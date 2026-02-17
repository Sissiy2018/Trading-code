from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
from scipy.stats import norm

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor


class SignalGenerator:
    def get_signals(self, **kwargs):
        raise NotImplementedError("Must implement get_signals")


class Momentum12_1M(SignalGenerator):
    def get_signals(self, hedged_returns):
        log_returns = np.log1p(hedged_returns)

        # Add min_periods! E.g., require at least 200 valid days out of 252
        mom_12m = log_returns.rolling(window=252, min_periods=200).sum()
        mom_1m = log_returns.rolling(window=21, min_periods=15).sum()

        signal = mom_12m - mom_1m
        return signal


# =============================================================================
# Config for the complex ML signal pipeline
# =============================================================================

@dataclass(frozen=True)
class SignalConfig:
    # Universe / liquidity filters
    adv_window: int = 60
    min_adv_dollars: float = 5e6
    min_price: float = 2.0
    min_coverage: float = 0.70

    # Trading limits
    adv_frac_limit: float = 0.025
    max_pos_usd: float = 2e6

    # Risk management
    risk_window: int = 60
    risk_limit_usd: float = 500_000.0
    risk_budget_usd: float = 400_000.0

    # Beta estimation
    beta_window: int = 250
    beta_shrink_a: float = 0.2
    beta_shrink_b: float = 0.8

    # Walk-forward model training
    train_window: int = 756
    val_window: int = 126
    refit_every: int = 21
    min_names: int = 100

    # Portfolio construction
    name_cap_weight: float = 0.02
    eta: float = 0.08
    max_turnover: float = 0.15

    # Signal / alpha parameters
    mom_horizons: Tuple[int, ...] = (21, 63, 126, 252)
    mom_weights: Tuple[float, ...] = (0.10, 0.25, 0.35, 0.30)
    mom_skip: int = 5
    mom_vol_window: int = 63
    mom_ema_alpha: float = 0.10

    strev_window: int = 5
    lowvol_window: int = 60
    high52_window: int = 252
    amihud_window: int = 60

    # Target variable
    fwd_return_days: int = 5

    # Robust cross-sectional transforms
    winsor_k: float = 5.0
    gauss_cap: float = 3.0


# =============================================================================
# Utility functions for the complex pipeline
# =============================================================================

def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def _cs_mad_winsorize(x: pd.DataFrame, k: float) -> pd.DataFrame:
    med = x.median(axis=1)
    mad = (x.sub(med, axis=0)).abs().median(axis=1)
    mad = mad.replace(0.0, np.nan)
    lo = med - k * 1.4826 * mad
    hi = med + k * 1.4826 * mad
    return x.clip(lower=lo, upper=hi, axis=0)


def _cs_rank_gauss(x: pd.DataFrame, cap: float, eps: float = 1e-6) -> pd.DataFrame:
    u = x.rank(axis=1, method="average", pct=True).clip(eps, 1 - eps)
    z = pd.DataFrame(norm.ppf(u), index=x.index, columns=x.columns)
    return z.clip(-cap, cap)


def _cs_robust(x: pd.DataFrame, winsor_k: float, gauss_cap: float) -> pd.DataFrame:
    return _cs_rank_gauss(_cs_mad_winsorize(x, winsor_k), gauss_cap)


def _safe_div(a: pd.DataFrame, b: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    return a / (b.replace(0.0, np.nan) + eps)


def _adv_dollars(prices: pd.DataFrame, volume: pd.DataFrame, window: int) -> pd.DataFrame:
    dv = (prices * volume).replace([np.inf, -np.inf], np.nan)
    return dv.rolling(window, min_periods=window).mean()


def _build_universe_mask(prices: pd.DataFrame, volume: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    px_ok = prices >= cfg.min_price
    adv = _adv_dollars(prices, volume, cfg.adv_window)
    adv_ok = adv >= cfg.min_adv_dollars
    coverage = prices.notna().rolling(cfg.adv_window, min_periods=cfg.adv_window).mean()
    cov_ok = coverage >= cfg.min_coverage
    return (px_ok & adv_ok & cov_ok)


def _position_caps_usd(prices: pd.DataFrame, volume: pd.DataFrame, cfg: SignalConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    adv = _adv_dollars(prices, volume, cfg.adv_window)
    trade_cap = cfg.adv_frac_limit * adv
    pos_cap = np.minimum(cfg.max_pos_usd, trade_cap)
    return pos_cap, trade_cap


def _rolling_beta(prices: pd.DataFrame, benchmark: pd.Series, cfg: SignalConfig) -> pd.DataFrame:
    r = _log_returns(prices)
    rb = _log_returns(benchmark.to_frame("b"))["b"].reindex(r.index)
    var_b = rb.rolling(cfg.beta_window, min_periods=cfg.beta_window).var(ddof=0)
    cov = r.rolling(cfg.beta_window, min_periods=cfg.beta_window).cov(rb)
    raw = cov.div(var_b, axis=0)
    return cfg.beta_shrink_a + cfg.beta_shrink_b * raw


def _sector_dummies(tickers: pd.Index, sectors: pd.Series) -> pd.DataFrame:
    s = sectors.reindex(tickers).fillna("UNKNOWN")
    return pd.get_dummies(s).reindex(index=tickers).fillna(0.0).astype(float)


def _cs_neutralize_panel(
    x: pd.DataFrame,
    sectors: pd.Series,
    beta: Optional[pd.DataFrame],
    mask: Optional[pd.DataFrame],
    ridge: float = 1e-6,
    min_names: int = 120,
) -> pd.DataFrame:
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    tickers = x.columns
    Dfull = _sector_dummies(tickers, sectors)

    for t in x.index:
        y = x.loc[t]
        if mask is not None:
            y = y.where(mask.loc[t])

        m = y.notna() & np.isfinite(y.values)
        if m.sum() < min_names:
            continue

        names = tickers[m.values]
        X_parts = [np.ones((len(names), 1)), Dfull.loc[names].values]
        if beta is not None:
            b = beta.loc[t].reindex(names).fillna(1.0).values.reshape(-1, 1)
            X_parts.append(b)

        X = np.concatenate(X_parts, axis=1).astype(float)
        yv = y.loc[names].values.astype(float)

        XtX = X.T @ X
        XtX.flat[:: XtX.shape[0] + 1] += ridge
        coef = np.linalg.solve(XtX, X.T @ yv)
        out.loc[t, names] = yv - X @ coef

    return out


def _residualize_daily_returns(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    sectors: pd.Series,
    mask: Optional[pd.DataFrame],
    min_names: int,
) -> pd.DataFrame:
    r = _log_returns(prices)
    rb = _log_returns(benchmark.to_frame("b"))["b"].reindex(r.index)
    tickers = r.columns
    Dfull = _sector_dummies(tickers, sectors)

    out = pd.DataFrame(index=r.index, columns=tickers, dtype=float)
    for t in r.index:
        y = r.loc[t]
        if mask is not None:
            y = y.where(mask.loc[t])
        m = y.notna() & np.isfinite(y.values)
        if m.sum() < min_names or pd.isna(rb.loc[t]):
            continue

        names = tickers[m.values]
        D = Dfull.loc[names].values
        yv = y.loc[names].values.astype(float)

        X = np.concatenate(
            [np.ones((len(names), 1)),
             np.full((len(names), 1), float(rb.loc[t])),
             D],
            axis=1
        )
        coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        out.loc[t, names] = yv - X @ coef
    return out


# =============================================================================
# Individual alpha signal functions
# =============================================================================

def _sig_momentum(resid_r: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    vol = resid_r.rolling(cfg.mom_vol_window, min_periods=cfg.mom_vol_window).std(ddof=0)
    comps = []
    for H, wH in zip(cfg.mom_horizons, cfg.mom_weights):
        mom = resid_r.shift(cfg.mom_skip).rolling(H, min_periods=H).sum()
        comps.append(wH * _cs_robust(_safe_div(mom, vol), cfg.winsor_k, cfg.gauss_cap))
    s = sum(comps).ewm(alpha=cfg.mom_ema_alpha, adjust=False).mean()
    return s


def _sig_st_reversal(resid_r: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    x = -(resid_r.rolling(cfg.strev_window, min_periods=cfg.strev_window).sum())
    return _cs_robust(x, cfg.winsor_k, cfg.gauss_cap)


def _sig_lowvol(prices: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    r = _log_returns(prices)
    v = r.rolling(cfg.lowvol_window, min_periods=cfg.lowvol_window).std(ddof=0)
    return _cs_robust(-v, cfg.winsor_k, cfg.gauss_cap)


def _sig_52w_high(prices: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    hi = prices.rolling(cfg.high52_window, min_periods=cfg.high52_window).max()
    x = prices / hi - 1.0
    return _cs_robust(x, cfg.winsor_k, cfg.gauss_cap)


def _sig_amihud(prices: pd.DataFrame, volume: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    r_abs = _log_returns(prices).abs()
    dv = (prices * volume).replace(0.0, np.nan)
    a = (r_abs / dv).rolling(cfg.amihud_window, min_periods=cfg.amihud_window).mean()
    return _cs_robust(a, cfg.winsor_k, cfg.gauss_cap)


def _sig_volume_momentum(volume: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    vol_short = volume.rolling(21, min_periods=21).mean()
    vol_long = volume.rolling(63, min_periods=63).mean()
    x = _safe_div(vol_short, vol_long) - 1.0
    return _cs_robust(x, cfg.winsor_k, cfg.gauss_cap)


def _sig_return_consistency(resid_r: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    pos_days = (resid_r > 0).astype(float)
    win_rate = pos_days.rolling(63, min_periods=63).mean()
    return _cs_robust(win_rate - 0.5, cfg.winsor_k, cfg.gauss_cap)


def _build_features(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    benchmark: pd.Series,
    sectors: pd.Series,
    fundamentals_panels: Optional[Dict[str, pd.DataFrame]],
    uni: pd.DataFrame,
    beta: pd.DataFrame,
    cfg: SignalConfig,
) -> Dict[str, pd.DataFrame]:
    resid_r = _residualize_daily_returns(prices, benchmark, sectors, mask=uni, min_names=cfg.min_names)

    feats: Dict[str, pd.DataFrame] = {}
    feats["mom"] = _sig_momentum(resid_r, cfg)
    feats["strev"] = _sig_st_reversal(resid_r, cfg)
    feats["lowvol"] = _sig_lowvol(prices, cfg)
    feats["high52"] = _sig_52w_high(prices, cfg)
    feats["amihud"] = _sig_amihud(prices, volume, cfg)
    feats["vol_mom"] = _sig_volume_momentum(volume, cfg)
    feats["win_rate"] = _sig_return_consistency(resid_r, cfg)

    if fundamentals_panels:
        if "btp" in fundamentals_panels:
            feats["value_btp"] = _cs_robust(fundamentals_panels["btp"], cfg.winsor_k, cfg.gauss_cap)
        if "prof" in fundamentals_panels:
            feats["quality_prof"] = _cs_robust(fundamentals_panels["prof"], cfg.winsor_k, cfg.gauss_cap)
        if "ag" in fundamentals_panels:
            feats["inv_ag"] = _cs_robust(-fundamentals_panels["ag"], cfg.winsor_k, cfg.gauss_cap)

    for k in list(feats.keys()):
        feats[k] = feats[k].reindex(index=prices.index, columns=prices.columns)
        feats[k] = _cs_neutralize_panel(feats[k], sectors=sectors, beta=beta, mask=uni, min_names=cfg.min_names)

    return feats


def _build_target(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    sectors: pd.Series,
    uni: pd.DataFrame,
    cfg: SignalConfig,
) -> pd.DataFrame:
    resid_r = _residualize_daily_returns(prices, benchmark, sectors, mask=uni, min_names=cfg.min_names)
    fwd = resid_r.rolling(cfg.fwd_return_days).sum().shift(-cfg.fwd_return_days)
    return fwd.reindex(prices.index)


# =============================================================================
# ML model and ensemble helpers
# =============================================================================

def _build_models(random_state: int = 0) -> Dict[str, Pipeline]:
    return {
        "ridge": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("m", Ridge(alpha=10.0)),
        ]),
        "ridge_heavy": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("m", Ridge(alpha=100.0)),
        ]),
        "hgb": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.03,
                max_leaf_nodes=15,
                min_samples_leaf=200,
                random_state=random_state,
            )),
        ]),
    }


def _panel_to_samples(
    X: Dict[str, pd.DataFrame],
    y: pd.DataFrame,
    dates: pd.DatetimeIndex,
    uni: pd.DataFrame,
    cfg: SignalConfig,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, list]:
    feat_names = list(X.keys())
    tickers = y.columns
    rows, targs, meta = [], [], []

    for t in dates:
        if t not in y.index:
            continue
        Xd = np.column_stack([X[f].loc[t].reindex(tickers).values for f in feat_names])
        yd = y.loc[t].reindex(tickers).values
        ok = np.isfinite(Xd).all(axis=1) & np.isfinite(yd)
        ok &= uni.loc[t].reindex(tickers).fillna(False).values
        if ok.sum() < cfg.min_names:
            continue
        rows.append(Xd[ok])
        targs.append(yd[ok])
        meta.extend([(t, tickers[i]) for i in np.where(ok)[0]])

    if not rows:
        raise ValueError("No samples built. Check lookbacks / universe / missingness.")

    return np.vstack(rows), np.concatenate(targs), pd.DataFrame(meta, columns=["date", "ticker"]), feat_names


def _predict_panel(
    model, X: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
    uni: pd.DataFrame, cfg: SignalConfig,
) -> pd.DataFrame:
    feat_names = list(X.keys())
    tickers = next(iter(X.values())).columns
    out = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for t in dates:
        Xd = np.column_stack([X[f].loc[t].reindex(tickers).values for f in feat_names])
        ok = np.isfinite(Xd).all(axis=1) & uni.loc[t].reindex(tickers).fillna(False).values
        if ok.sum() < cfg.min_names:
            continue
        out.loc[t, tickers[ok]] = model.predict(Xd[ok])
    return out


def _daily_spearman_ic(pred: pd.DataFrame, y: pd.DataFrame, min_names: int) -> pd.Series:
    idx = pred.index.intersection(y.index)
    out = []
    for t in idx:
        a, b = pred.loc[t], y.loc[t]
        m = a.notna() & b.notna() & np.isfinite(a.values) & np.isfinite(b.values)
        if m.sum() < min_names:
            out.append(np.nan)
            continue
        out.append(a[m].rank().corr(b[m].rank()))
    return pd.Series(out, index=idx)


def _ic_stats(ic: pd.Series) -> Tuple[float, float, float]:
    mu = ic.mean(skipna=True)
    sd = ic.std(skipna=True, ddof=0)
    ir = mu / sd if (sd and np.isfinite(sd) and sd > 0) else np.nan
    return float(mu), float(sd), float(ir)


def _ensemble_weights_from_val(d: Dict[str, dict]) -> Dict[str, float]:
    raw = {}
    for name, info in d.items():
        mu, sd, ir = info["mean_ic"], info["std_ic"], info["ir"]
        score = ir if np.isfinite(ir) else mu
        raw[name] = max(0.0, float(score)) if np.isfinite(score) else 0.0
    s = sum(raw.values())
    if s <= 0:
        n = len(raw)
        return {k: 1.0 / n for k in raw}
    return {k: v / s for k, v in raw.items()}


def _combine_predictions(preds: Dict[str, pd.DataFrame], w: Dict[str, float]) -> pd.DataFrame:
    out = None
    for k, pk in preds.items():
        wk = w.get(k, 0.0)
        if wk == 0:
            continue
        out = pk * wk if out is None else out.add(pk * wk, fill_value=np.nan)
    return out


# =============================================================================
# ComplexMLSignal: walk-forward ML ensemble signal generator
# =============================================================================

class ComplexMLSignal(SignalGenerator):
    """Walk-forward ML ensemble signal that produces alpha scores.

    Builds 7 technical features (momentum, reversal, low-vol, 52w-high,
    Amihud, volume momentum, win-rate), trains a Ridge + HGB ensemble
    via walk-forward validation, and returns sector/beta-neutralised
    alpha predictions as a DataFrame (dates x tickers).

    Usage in wrapper.py:
        signal_gen = ComplexMLSignal()
        signals = signal_gen.get_signals(
            hedged_returns,
            prices=price_close_imp,
            volume=volume_usd,
            benchmark=benchmark_series,
            sectors=sectors_series,
        )
    """

    def __init__(self, cfg: Optional[SignalConfig] = None, random_state: int = 42):
        self.cfg = cfg or SignalConfig()
        self.random_state = random_state
        self.diagnostics_ = {}

    def get_signals(
        self,
        hedged_returns,
        prices=None,
        volume=None,
        benchmark=None,
        sectors=None,
        fundamentals_panels=None,
    ):
        cfg = self.cfg

        prices = prices.sort_index()
        volume = volume.reindex_like(prices).sort_index()
        benchmark = benchmark.reindex(prices.index).sort_index()
        sectors = sectors.reindex(prices.columns)

        # Step 1: Universe + caps
        uni = _build_universe_mask(prices, volume, cfg)

        # Step 2: Rolling beta
        beta = _rolling_beta(prices, benchmark, cfg)

        # Step 3: Features and target
        X = _build_features(prices, volume, benchmark, sectors, fundamentals_panels, uni, beta, cfg)
        y = _build_target(prices, benchmark, sectors, uni, cfg)

        models = _build_models(random_state=self.random_state)

        dates = prices.index
        tickers = prices.columns
        pred_all = pd.DataFrame(index=dates, columns=tickers, dtype=float)
        ensemble_hist = {}
        ic_hist = {}

        # Step 4: Walk-forward schedule
        start = max(cfg.train_window, cfg.adv_window, cfg.beta_window,
                     cfg.mom_vol_window, cfg.high52_window) + 5
        refits = list(range(start, len(dates), cfg.refit_every))

        for rp in refits:
            t_refit = dates[rp]

            tr0 = dates[max(0, rp - cfg.train_window)]
            va0 = dates[rp]
            va1 = dates[min(len(dates) - 1, rp + cfg.val_window)]

            train_dates = dates[(dates >= tr0) & (dates < va0)]
            val_dates = dates[(dates >= va0) & (dates <= va1)]
            if len(train_dates) < 50 or len(val_dates) < 20:
                continue

            Xtr, ytr, _, _ = _panel_to_samples(X, y, train_dates, uni, cfg)

            diag = {}
            for name, mdl in models.items():
                mdl.fit(Xtr, ytr)
                pv = _predict_panel(mdl, X, val_dates, uni, cfg)
                ic = _daily_spearman_ic(pv, y.loc[val_dates], min_names=cfg.min_names)
                mu, sd, ir = _ic_stats(ic)
                diag[name] = {"ic": ic, "mean_ic": mu, "std_ic": sd, "ir": ir}

            w_ens = _ensemble_weights_from_val(diag)
            ensemble_hist[t_refit] = w_ens
            ic_hist[t_refit] = {k: (v["mean_ic"], v["ir"]) for k, v in diag.items()}

            # Refit on train + validation combined, then predict the next block
            fit_dates = dates[(dates >= tr0) & (dates <= va1)]
            Xfit, yfit, _, _ = _panel_to_samples(X, y, fit_dates, uni, cfg)

            preds_fwd = {}
            nxt = min(len(dates), rp + cfg.refit_every)
            block_dates = dates[rp:nxt]
            for name, mdl in models.items():
                mdl.fit(Xfit, yfit)
                preds_fwd[name] = _predict_panel(mdl, X, block_dates, uni, cfg)

            pred_block = _combine_predictions(preds_fwd, w_ens)
            pred_all.loc[pred_block.index] = pred_block

        # Step 5: Post-process alpha
        alpha = _cs_robust(pred_all, cfg.winsor_k, cfg.gauss_cap)
        alpha_n = _cs_neutralize_panel(alpha, sectors=sectors, beta=beta,
                                        mask=uni, min_names=cfg.min_names)

        # Store diagnostics and intermediate objects for the portfolio constructor
        self.diagnostics_ = {
            "ensemble_weights_by_refit": ensemble_hist,
            "val_ic_by_refit": ic_hist,
        }
        self.uni_ = uni
        self.beta_ = beta
        self.pos_cap_, self.trade_cap_ = _position_caps_usd(prices, volume, cfg)
        self.sectors_ = sectors
        self.prices_ = prices

        return alpha_n
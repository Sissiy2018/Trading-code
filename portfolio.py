import pandas as pd
import numpy as np

class PortfolioConstructor:
    def __init__(self, target_ann_vol=500000, max_adv_pct=0.025):
        self.target_daily_vol = target_ann_vol / np.sqrt(252)
        self.max_adv_pct = max_adv_pct

    def generate_target_positions(self, t, signals, cov_matrix, adv_60d, betas, benchmark_ticker):
        sig_t = signals.dropna()
        if len(sig_t) < 10:
            return pd.Series(0.0, index=signals.index)
            
        top_n = max(1, int(len(sig_t) * 0.10))
        longs = sig_t.nlargest(top_n).index
        shorts = sig_t.nsmallest(top_n).index

        # --- NEW: Signal-Weighted Risk Weighting ---
        vols = np.sqrt(np.diag(cov_matrix.loc[sig_t.index, sig_t.index]))
        vol_series = pd.Series(vols, index=sig_t.index)
        
        raw_weights = pd.Series(0.0, index=sig_t.index)
        
        # Multiply inverse volatility by the absolute signal strength
        long_signal_risk = sig_t[longs].abs() / vol_series[longs]
        short_signal_risk = sig_t[shorts].abs() / vol_series[shorts]
        
        # Normalize so longs sum to 100% and shorts sum to -100% (before vol scaling)
        raw_weights[longs] = long_signal_risk / long_signal_risk.sum()
        raw_weights[shorts] = -short_signal_risk / short_signal_risk.sum()
        # ------------------------------------------

        # Initial Volatility Scaling
        port_vol = np.sqrt(raw_weights.T @ cov_matrix.loc[sig_t.index, sig_t.index] @ raw_weights)
        scalar = self.target_daily_vol / port_vol if port_vol > 0 else 0
        pos = raw_weights * scalar 

        # Iterative Liquidity Constraints & Redistribution
        max_pos = adv_60d.loc[sig_t.index] * self.max_adv_pct
        #make sure max_pos dont excceed 2000000 in size
        max_pos = max_pos.clip(upper=2000000, lower=-2000000)
        unconstrained = set(longs).union(set(shorts))
        
        for _ in range(10): # Max iterations to prevent infinite loops
            breaches = False
            for asset in list(unconstrained):
                if abs(pos[asset]) > max_pos[asset]:
                    pos[asset] = np.sign(pos[asset]) * max_pos[asset]
                    unconstrained.remove(asset)
                    breaches = True
            
            if not breaches or not unconstrained:
                break # All constraints satisfied
                
            # If there were breaches, recalculate vol and scale up unconstrained
            active = pos[pos != 0].index
            current_vol = np.sqrt(pos[active].T @ cov_matrix.loc[active, active] @ pos[active])
            
            if current_vol >= self.target_daily_vol or current_vol == 0:
                break
                
            # Scale up unconstrained assets to bridge the gap
            rescale_factor = self.target_daily_vol / current_vol
            for asset in unconstrained:
                pos[asset] *= rescale_factor

        # Calculate Benchmark Hedge Position
        assets_only = pos.index.difference([benchmark_ticker])
        beta_exposure = (pos[assets_only] * betas[assets_only]).sum()
        pos[benchmark_ticker] = -beta_exposure

        return pos


class ComplexPortfolioConstructor(PortfolioConstructor):
    """Portfolio constructor that converts complex ML alpha scores into
    dollar-neutral positions with per-name caps and liquidity constraints.

    Designed to work with ComplexMLSignal: receives per-day alpha scores
    via the same generate_target_positions interface as PortfolioConstructor.

    Key differences from the base class:
      - Uses ALL stocks with a signal (not top/bottom 10%)
      - Weights are proportional to alpha score (demeaned, normalised)
      - Per-name weight cap (default 2%) prevents concentration
      - Position cap = min(max_pos_usd, max_adv_pct * ADV)
      - Beta hedge via benchmark
    """

    def __init__(self, target_ann_vol=500000, max_adv_pct=0.025, max_pos_usd=2e6,
                 name_cap_weight=0.02):
        super().__init__(target_ann_vol, max_adv_pct)
        self.max_pos_usd = max_pos_usd
        self.name_cap_weight = name_cap_weight

    def generate_target_positions(self, t, signals, cov_matrix, adv_60d, betas, benchmark_ticker):
        sig_t = signals.dropna()
        sig_t = sig_t[np.isfinite(sig_t)]
        if len(sig_t) < 10:
            return pd.Series(0.0, index=signals.index)

        # Dollar-neutral weights: demean then normalise to unit gross
        w = sig_t - sig_t.mean()
        gross = w.abs().sum()
        if gross == 0 or not np.isfinite(gross):
            return pd.Series(0.0, index=signals.index)
        w = w / gross

        # Apply per-name weight cap and re-normalise
        w = w.clip(-self.name_cap_weight, self.name_cap_weight)
        w = w - w.mean()
        gross = w.abs().sum()
        if gross == 0 or not np.isfinite(gross):
            return pd.Series(0.0, index=signals.index)
        w = w / gross

        # Scale weights to dollar notionals targeting the risk budget
        # Use covariance to estimate portfolio vol, then scale
        valid = w.index.intersection(cov_matrix.index).intersection(cov_matrix.columns)
        w_valid = w.reindex(valid).fillna(0.0)
        cov_valid = cov_matrix.loc[valid, valid]
        port_vol = np.sqrt(w_valid.values @ cov_valid.values @ w_valid.values)
        scalar = self.target_daily_vol / port_vol if (port_vol > 0 and np.isfinite(port_vol)) else 0
        pos = w * scalar

        # Apply per-stock position cap: min(max_pos_usd, max_adv_pct * ADV)
        max_pos = (adv_60d.reindex(sig_t.index).fillna(0.0) * self.max_adv_pct).clip(upper=self.max_pos_usd)
        pos = pos.clip(lower=-max_pos, upper=max_pos)

        # Re-pad to full universe
        full_pos = pd.Series(0.0, index=signals.index)
        full_pos.update(pos)

        # Beta hedge via benchmark
        assets_only = full_pos.index.difference([benchmark_ticker])
        beta_t = betas.reindex(assets_only).fillna(1.0)
        beta_exposure = (full_pos[assets_only] * beta_t).sum()
        full_pos[benchmark_ticker] = -beta_exposure

        return full_pos
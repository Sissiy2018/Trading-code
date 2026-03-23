from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import norm

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import TruncatedSVD
from hmmlearn import hmm
import warnings

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

def robust_cross_sectional_norm(sig_df: pd.DataFrame, limit: float = 3.0) -> pd.DataFrame:
    cs_median = sig_df.median(axis=1)
    
    # 1. Calculate MAD and standard deviation
    cs_mad = (sig_df.sub(cs_median, axis=0)).abs().median(axis=1) * 1.4826
    cs_std = sig_df.std(axis=1)
    
    # 2. Safe Denominator: If MAD is 0, fall back to a fraction of the Std Dev
    safe_denom = np.maximum(cs_mad, cs_std * 0.2).replace(0, 1e-6)
    
    # 3. Z-Score
    norm_z = sig_df.sub(cs_median, axis=0).div(safe_denom, axis=0)
    
    # 4. SOFT CLIPPING (The secret sauce to prevent discretized walls)
    # This smoothly asymptotes to +/- limit without bunching up outliers
    soft_clipped = limit * np.tanh(norm_z / limit)
    
    return soft_clipped.fillna(0.0)



class ShortTermSignalGenerator:
    """Generates fast, mean-reversion signals based on short-term price action."""
    def __init__(self, reversal_window=10, smoothing_span=2):
        self.window = reversal_window
        self.span = smoothing_span

    def generate(self, hedged_returns):
        print(f"  -> Generating Short-Term Signals ({self.window}d Reversal)...")
        # 1. Calculate short-term returns
        ret_short = hedged_returns.rolling(self.window, min_periods=self.window-2).sum()
        
        # 2. Cross-sectional Z-score
        cs_mean = ret_short.mean(axis=1)
        cs_std = ret_short.std(axis=1)
        z_score = ret_short.sub(cs_mean, axis=0).div(cs_std + 1e-8, axis=0)
        
        # 3. Invert for Mean Reversion (Buy losers, sell winners)
        signal = z_score
        
        # Smooth the raw signals if you aren't already
        sig_df = signal.ewm(span=self.span, min_periods=1).mean()

        # SWAP TO ROBUST NORMALIZATION
        final_z = robust_cross_sectional_norm(sig_df)
        
        return final_z


class VolumeConvictionGenerator:
    """
    Generates an orthogonal signal based on the Institutional Footprint.
    It identifies high-conviction moves by weighting idiosyncratic returns 
    by their relative volume shock.
    """
    def __init__(self, volume_window=10, return_window=3, smoothing_span=3):
        self.volume_window = volume_window
        self.return_window = return_window
        self.smoothing_span = smoothing_span

    def generate(self, hedged_returns, volume):
        # 1. Calculate the Volume Shock (Ratio of today's volume to moving average)
        # Add 1e-8 to avoid division by zero on halted/stale tickers
        avg_volume = volume.rolling(window=self.volume_window, min_periods=10).mean()
        volume_shock = volume / (avg_volume + 1e-8)
        
        # 2. Calculate directional conviction
        # We look at a short rolling return to capture the immediate trend direction
        trend_ret = hedged_returns.rolling(window=self.return_window, min_periods=3).sum()
        
        # 3. The raw signal: Direction * Magnitude of Volume Shock
        # High volume up-moves = highly positive. High volume down-moves = highly negative.
        raw_conviction = trend_ret * volume_shock
        
        # 4. Cross-sectional z-score daily to neutralize market-wide volume events
        mean_conviction = raw_conviction.mean(axis=1)
        std_conviction = raw_conviction.std(axis=1) + 1e-8
        z_scored = raw_conviction.sub(mean_conviction, axis=0).div(std_conviction, axis=0)
        
        # 5. Smooth the signal to reduce daily turnover and trading costs
        smoothed_signal = z_scored.ewm(span=self.smoothing_span, min_periods=1).mean()
        
        return smoothed_signal


class LongTermSignalGenerator:
    """
    Generates long-term signals using Momentum as the primary engine, 
    but strictly uses Sector-Neutral Value as a 'Guardrail' multiplier to 
    penalize expensive bubbles and reward cheap compounders.
    """
    def __init__(self, momentum_window=252, skip_recent=21, smoothing_span=10, value_tilt_strength=0.25):
        self.window = momentum_window
        self.skip = skip_recent
        self.span = smoothing_span
        self.tilt = value_tilt_strength 

    def generate(self, hedged_returns, earnings_yield, sectors):
        print(f"  -> Generating Long-Term Signals (Value-Tilted Momentum)...")
        dates = hedged_returns.index
        tickers = hedged_returns.columns
        
        # --- 1. Momentum Component (The Core Engine) ---
        ret_long = hedged_returns.rolling(self.window - self.skip).sum().shift(self.skip)
        # UPDATE: Robustly normalize the raw momentum returns
        mom_z = robust_cross_sectional_norm(ret_long)
        
        # --- 2. Value Component (The Guardrail) ---
        ey_aligned = earnings_yield.ffill().fillna(0)
        val_z = pd.DataFrame(0.0, index=dates, columns=tickers)
        
        unique_sectors = sectors.unique()
        for sec in unique_sectors:
            sec_tickers = sectors[sectors == sec].index.intersection(tickers)
            if len(sec_tickers) > 1:
                sec_ey = ey_aligned[sec_tickers]
                # UPDATE: Robustly normalize the earnings yield within the sector
                sec_z = robust_cross_sectional_norm(sec_ey)
                val_z[sec_tickers] = sec_z
                
        val_z = val_z.fillna(0)
        
        # --- 3. The Conviction Multiplier (Robust Blend) ---
        # Clip value to strictly prevent extreme data outliers from breaking the signal
        # (We keep this at 2.0 based on your original logic to prevent the multiplier from getting too extreme)
        safe_val = val_z.clip(lower=-2.0, upper=2.0)
        
        value_multiplier = 1.0 + (safe_val * self.tilt)
        tilted_mom = mom_z * value_multiplier
        
        # --- 4. Smooth and Re-Normalize ---
        # UPDATE: Smooth FIRST, then apply the final robust normalization. 
        # This guarantees the output perfectly matches the bounds of your PCA and Short-Term generators.
        smoothed_tilted = tilted_mom.ewm(span=self.span, min_periods=1).mean()
        final_z = robust_cross_sectional_norm(smoothed_tilted)
        
        return final_z



class PCASignalGenerator:
    """
    Extracts the top N principal components from the rolling covariance matrix.
    Uses TruncatedSVD for a massive O(N^3) speedup and MAD normalization for stability.
    """
    def __init__(self, n_components=20, cov_window=252, mom_window=126, rev_window=21, span=10, pca_update_freq=5):
        self.k = n_components
        self.cov_win = cov_window
        self.mom_win = mom_window
        self.rev_win = rev_window
        self.span = span
        self.update_freq = pca_update_freq 

    def generate(self, returns):
        print(f"  -> Generating PCA Signals (Top {self.k} Factors, Updating PCA every {self.update_freq} days)...")
        
        ret_mom = returns.rolling(self.mom_win).sum()
        ret_rev = returns.rolling(self.rev_win).sum()
        
        n_days, n_assets = returns.shape
        pca_signals = np.zeros((n_days, n_assets))
        
        ret_vals = returns.fillna(0.0).values
        mom_vals = ret_mom.fillna(0.0).values
        rev_vals = ret_rev.fillna(0.0).values
        
        top_vecs = None 
        
        # Instantiate the solver once
        svd = TruncatedSVD(n_components=self.k, n_iter=5, random_state=42)
        
        for i in range(self.cov_win, n_days):
            
            # --- EXPENSIVE STEP: Now Lightning Fast ---
            if i % self.update_freq == 0 or top_vecs is None:
                window_data = ret_vals[i - self.cov_win : i]
                
                # TruncatedSVD requires mean-centered data to act as true PCA
                window_data_centered = window_data - np.mean(window_data, axis=0)
                
                # Extract top K components without building the N x N covariance matrix
                svd.fit(window_data_centered)
                top_vecs = svd.components_.T # Shape: (n_assets, k)
            
            # --- CHEAP STEP: Run every day ---
            comp_mom = mom_vals[i] @ top_vecs
            comp_rev = rev_vals[i] @ top_vecs
            
            # Safe standardization of the projected components
            mom_std = np.std(comp_mom)
            rev_std = np.std(comp_rev)
            
            comp_mom_z = (comp_mom - np.mean(comp_mom)) / mom_std if mom_std > 1e-8 else np.zeros(self.k)
            comp_rev_z = -(comp_rev - np.mean(comp_rev)) / rev_std if rev_std > 1e-8 else np.zeros(self.k)
                
            comp_signal = 0.5 * comp_mom_z + 0.5 * comp_rev_z
            pca_signals[i] = top_vecs @ comp_signal
            
        sig_df = pd.DataFrame(pca_signals, index=returns.index, columns=returns.columns)
        sig_df = sig_df.ewm(span=self.span, min_periods=1).mean()
        
        # --- ROBUST NORMALIZATION ---
        # 1. Use Cross-Sectional Median instead of Mean
        cs_median = sig_df.median(axis=1)
        
        # 2. Compute Median Absolute Deviation (MAD)
        cs_mad = (sig_df.sub(cs_median, axis=0)).abs().median(axis=1)
        
        # 3. Scale using MAD (1.4826 makes it asymptotically equal to standard deviation)
        final_z = sig_df.sub(cs_median, axis=0).div(cs_mad * 1.4826 + 1e-8, axis=0)
        
        # 4. Strict Winsorization: Clip extreme outliers that bypass the MAD buffer
        final_z = final_z.clip(lower=-3.0, upper=3.0)
        
        return final_z.fillna(0)


# Suppress hmmlearn warnings about covariance regularization

class RegimePCAHMMGenerator:
    """
    Regime-switching PCA factor model. Uses TruncatedSVD for fast factor extraction 
    and a Gaussian HMM to model the latent states of the factor returns.
    """
    def __init__(self, n_components=8, pca_update_freq=42, initial_states=2, max_states=5, hmm_window=500):
        self.k = n_components
        self.freq = pca_update_freq 
        self.n_states = initial_states
        self.max_states = max_states
        self.hmm_window = hmm_window 
        self.model = None

    def _calc_bic(self, model, X):
        try:
            log_likelihood = model.score(X)
            n_features = X.shape[1]
            n_states = model.n_components
            n_params = n_states * (n_states - 1) + 2 * n_features * n_states
            bic = -2 * log_likelihood + n_params * np.log(X.shape[0])
            return bic
        except Exception:
            return np.inf

    def _fit_best_hmm(self, X, current_states):
        best_bic = np.inf
        best_model = None
        best_states = current_states
        
        # 1. Safely test the current state configuration
        try:
            model_curr = hmm.GaussianHMM(n_components=current_states, covariance_type="diag", n_iter=100, random_state=42)
            model_curr.fit(X)
            bic_curr = self._calc_bic(model_curr, X)
            
            if bic_curr != np.inf:
                best_model = model_curr
                best_bic = bic_curr
                best_states = current_states
        except Exception:
            pass # Fails gracefully
        
        # 2. Safely test scaling up by 1 state
        if current_states < self.max_states:
            try:
                model_up = hmm.GaussianHMM(n_components=current_states + 1, covariance_type="diag", n_iter=100, random_state=42)
                model_up.fit(X)
                bic_up = self._calc_bic(model_up, X)
                
                if bic_up < best_bic:
                    best_model = model_up
                    best_bic = bic_up
                    best_states = current_states + 1
            except Exception:
                pass
                
        # 3. BULLETPROOF FALLBACK: If both failed due to data singularities
        if best_model is None:
            best_model = hmm.GaussianHMM(n_components=2, covariance_type="diag", init_params="")
            # Manually inject safe, uniform parameters so it never crashes
            best_model.startprob_ = np.array([0.5, 0.5])
            best_model.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
            best_model.means_ = np.zeros((2, X.shape[1]))
            best_model.covars_ = np.ones((2, X.shape[1]))
            best_states = 2
                
        return best_model, best_states

    def generate(self, returns):
        print(f"  -> Generating HMM-PCA Signals (Top {self.k} PCs, updating states every {self.freq} days)...")
        
        n_days, n_assets = returns.shape
        signal_matrix = np.zeros((n_days, n_assets))
        
        ret_vals = returns.fillna(0.0).values
        
        svd = TruncatedSVD(n_components=self.k, random_state=42)
        top_vecs = None
        pc_returns = np.zeros((n_days, self.k))
        
        burn_in = max(60, self.k * 3) 
        
        for i in range(1, n_days):
            
            if i < burn_in:
                continue 
            
            if i % self.freq == 0 or top_vecs is None:
                lookback_start = max(0, i - self.hmm_window)
                window_data = ret_vals[lookback_start:i]
                
                window_data_centered = window_data - np.mean(window_data, axis=0)
                svd.fit(window_data_centered)
                top_vecs = svd.components_.T 
                
                pc_hist = window_data_centered @ top_vecs
                pc_returns[lookback_start:i] = pc_hist
                
                self.model, self.n_states = self._fit_best_hmm(pc_hist, self.n_states)
                
            else:
                today_centered = ret_vals[i] - np.mean(ret_vals[i])
                pc_returns[i] = today_centered @ top_vecs
                
                lookback_start = max(0, i - self.hmm_window)
                X_recent = pc_returns[lookback_start : i+1]
                
                # --- DAILY UPDATE PROTECTION ---
                try:
                    self.model.init_params = '' 
                    self.model.n_iter = 5
                    self.model.fit(X_recent)
                except Exception:
                    # If daily fitting hits a singularity, ignore it and keep yesterday's stable parameters
                    pass 
                
            # --- SIGNAL GENERATION ---
            lookback_start = max(0, i - self.hmm_window)
            
            # Sanitizer block (keeps probabilities mathematically valid)
            if hasattr(self.model, 'transmat_'):
                row_sums = self.model.transmat_.sum(axis=1)
                for r_idx, r_sum in enumerate(row_sums):
                    if np.isclose(r_sum, 0.0) or np.isnan(r_sum):
                        self.model.transmat_[r_idx, :] = 1.0 / self.model.n_components
                    else:
                        self.model.transmat_[r_idx, :] /= r_sum
                        
            if hasattr(self.model, 'startprob_'):
                s_sum = np.sum(self.model.startprob_)
                if np.isclose(s_sum, 0.0) or np.isnan(s_sum):
                    self.model.startprob_ = np.ones(self.model.n_components) / self.model.n_components
                else:
                    self.model.startprob_ /= s_sum
            
            # Safely predict probabilities
            try:
                filtered_probs = self.model.predict_proba(pc_returns[lookback_start : i+1])
                curr_state_prob = filtered_probs[-1] 
            except Exception:
                curr_state_prob = np.ones(self.model.n_components) / self.model.n_components
            
            tomorrows_prob = curr_state_prob @ self.model.transmat_ 
            
            E_R = np.zeros(n_assets)
            Var_R = np.zeros(n_assets)
            
            for s in range(self.n_states):
                mu_s_asset = top_vecs @ self.model.means_[s] 
                E_R += tomorrows_prob[s] * mu_s_asset
                
                Sigma_s = self.model.covars_[s]
                var_s_asset = np.sum(top_vecs * (top_vecs @ Sigma_s.T), axis=1)
                
                Var_R += tomorrows_prob[s] * (var_s_asset + (mu_s_asset)**2)
                
            Var_R = Var_R - E_R**2
            
            daily_sharpe = E_R / (np.sqrt(Var_R) + 1e-8)
            signal_matrix[i] = daily_sharpe
            
        sig_df = pd.DataFrame(signal_matrix, index=returns.index, columns=returns.columns)
        final_z = robust_cross_sectional_norm(sig_df)
        
        return final_z


class RobustRegressionBlender:
    """
    Uses Fama-MacBeth style Huber Regression with L2 penalty to extract daily signal 
    predictiveness. Applies a Temperature-Scaled Softmax to the smoothed coefficients.
    Supports an arbitrary number of signals and an optional prior weight distribution.
    """
    def __init__(self, lookback=60, temperature=3.0):
        self.lookback = lookback
        self.temperature = temperature 

    def blend(self, signals_dict, hedged_returns, prior_weights=None):
        """
        Args:
            signals_dict (dict): Dictionary mapping string names to signal DataFrames.
                                 e.g., {'short': df1, 'long': df2, 'pca': df3}
            hedged_returns (pd.DataFrame): The target returns to regress against.
            prior_weights (list/array, optional): Prior distribution for the weights. 
                                                  Defaults to uniform if None.
        """
        signal_names = list(signals_dict.keys())
        n_signals = len(signal_names)
        
        print(f"  -> Blending {n_signals} signals via Softmax Robust Regression (Temp: {self.temperature})...")
        
        # 1. Handle Prior Weights
        if prior_weights is None:
            # Default to uniform prior (1/N)
            prior_weights = np.ones(n_signals) / n_signals
        else:
            prior_weights = np.array(prior_weights, dtype=float)
            if len(prior_weights) != n_signals:
                raise ValueError(f"Length of prior_weights ({len(prior_weights)}) must match number of signals ({n_signals}).")
            # Normalize just in case they don't sum to exactly 1.0
            prior_weights = prior_weights / np.sum(prior_weights)
            
        # Create a Pandas Series for easy broadcasting later
        prior_series = pd.Series(prior_weights, index=signal_names)
        
        # We add log(prior) to the logits. Add 1e-12 to avoid log(0) if a prior is strictly 0.
        log_prior = np.log(prior_series + 1e-12)

        n_days, n_assets = hedged_returns.shape
        daily_weights = np.zeros((n_days, n_signals))
        
        ret_vals = hedged_returns.values
        # Extract underlying numpy arrays for all signals for fast iteration
        sig_vals = [signals_dict[name].values for name in signal_names]
        
        huber = HuberRegressor(fit_intercept=False, alpha=1.0, max_iter=100)
        
        # 2. Daily Regression Loop
        for i in range(1, n_days):
            y = ret_vals[i]
            # Dynamically stack the (i-1)th row of every signal into our X matrix
            X = np.column_stack([vals[i-1] for vals in sig_vals])
            
            valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
            
            if valid_mask.sum() > 50: 
                try:
                    huber.fit(X[valid_mask], y[valid_mask])
                    daily_weights[i] = huber.coef_
                except Exception:
                    pass
                    
        # 3. Process Coefficients
        alpha_shrinkage = 0.25
        w_df = pd.DataFrame(daily_weights, index=hedged_returns.index, columns=signal_names)
        smoothed_coefs = w_df.rolling(self.lookback, min_periods=10).mean()
        
        # --- NEW: ALPHA COVARIANCE ADJUSTMENT ---
        print(f"  -> Applying Cross-Sectional Alpha Orthogonalization...")
        adjusted_coefs = smoothed_coefs.copy()
        
        for i, date in enumerate(hedged_returns.index):
            if i < 10 or smoothed_coefs.loc[date].isna().all():
                continue
                
            # 1. Build the cross-section of today's signals (n_assets x n_signals)
            today_signals = pd.DataFrame({name: signals_dict[name].loc[date] for name in signal_names})
            today_signals = today_signals.dropna()
            
            if len(today_signals) < 50:
                continue
                
            # 2. Calculate the cross-sectional correlation matrix of the signals
            # This tells us how redundant the signals are today
            sig_corr = today_signals.corr().fillna(0).values
            
            # 3. Shrinkage (just like asset covariance) to ensure it's invertible
            identity_mat = np.eye(n_signals)
            robust_corr = (1.0 - alpha_shrinkage) * sig_corr + (alpha_shrinkage * identity_mat)
            
            # 4. Calculate the Inverse Correlation Matrix
            try:
                inv_corr = np.linalg.inv(robust_corr)
            except np.linalg.LinAlgError:
                inv_corr = identity_mat # Fallback if perfectly collinear
                
            # 5. Apply the adjustment: W_optimal = Inverse_Covariance @ W_raw
            raw_w = smoothed_coefs.loc[date].values
            optimal_w = inv_corr @ raw_w
            
            adjusted_coefs.loc[date] = optimal_w
            
        # --- RESUME ORIGINAL LOGIC ---
        scale_factor = adjusted_coefs.abs().mean(axis=1) + 1e-8
        scaled_coefs = adjusted_coefs.div(scale_factor, axis=0)
        
        # 4. Apply Temperature-Scaled Softmax with Bayesian Prior
        scaled_z = scaled_coefs.div(self.temperature)
        
        # Add the log_prior to mathematically shift the base probabilities
        scaled_z = scaled_z.add(log_prior, axis=1) 
        
        exp_z = np.exp(scaled_z.sub(scaled_z.max(axis=1), axis=0)) # Subtract max for stability
        softmax_weights = exp_z.div(exp_z.sum(axis=1), axis=0)
        
        # Fill burn-in NaN periods with the base prior weights
        softmax_weights = softmax_weights.fillna(prior_series)
        self.historical_weights = softmax_weights
        
        # 5. Apply dynamic Softmax weights to today's signals
        # Initialize an empty DataFrame of zeros to accumulate the blend
        blended = pd.DataFrame(0.0, index=hedged_returns.index, columns=hedged_returns.columns)
        
        for name in signal_names:
            blended += signals_dict[name].mul(softmax_weights[name], axis=0)
                   
        return blended

class DefensiveSignalGenerator:
    """
    Generates slow-moving, defensive signals designed to survive bear markets and rate-hike regimes.
    Combines 'Betting Against Beta' (Low Beta), Low Volatility, and 1-Month Earnings Drift.
    """
    def __init__(self, drift_window=21, vol_window=63, smoothing_span=10):
        self.drift_window = drift_window
        self.vol_window = vol_window
        self.span = smoothing_span

    def generate(self, hedged_returns, betas):
        print(f"  -> Generating Defensive Signals (Low Vol, Low Beta, 1M Drift)...")
        
        # --- 1. Low Volatility Anomaly ---
        # Calculate 3-month rolling volatility. We invert it (multiply by -1) 
        # so we are Long Low-Vol and Short High-Vol.
        rolling_vol = hedged_returns.rolling(self.vol_window, min_periods=self.vol_window-10).std()
        low_vol_z = robust_cross_sectional_norm(-rolling_vol)
        
        # --- 2. Betting Against Beta (BAB) ---
        # We want to be Long Low Beta, Short High Beta. 
        # (Assuming 'betas' is aligned with your returns DataFrame)
        bab_z = robust_cross_sectional_norm(-betas.fillna(1.0))
        
        # --- 3. 1-Month Earnings Drift (Intermediate Momentum) ---
        # 21-day trend continuation.
        drift_ret = hedged_returns.rolling(self.drift_window, min_periods=10).sum()
        drift_z = robust_cross_sectional_norm(drift_ret)
        
        # --- Blend the Defensive Traits ---
        # We equal-weight the three defensive properties. 
        # This creates a highly stable, slow-moving signal.
        defensive_blend = (0.4 * low_vol_z) + (0.4 * bab_z) + (0.2 * drift_z)
        
        # Heavy smoothing to ensure this acts as a low-turnover anchor
        smoothed_defensive = defensive_blend.ewm(span=self.span, min_periods=1).mean()
        
        final_z = robust_cross_sectional_norm(smoothed_defensive)
        return final_z


class EPSForecastSignalGenerator:
    """
    EPS-based forecast signal with weighted components:
    1) Dispersion: Price / EPS Std
    2) EPS Revision: pct_change(EPS Mean, revision_window)
    3) EPS Predicted Surprise PCT (optional)

    Final alpha (cross-sectionally normalized):
    alpha = -w_dispersion * dispersion_z + w_revision * revision_z + pct_sign * w_pct * pct_z
    """

    def __init__(
        self,
        revision_window=21,
        revision_smooth_window=1,
        smoothing_span=5,
        dispersion_window=1,
        pct_smooth_window=1,
        clip_dispersion=10.0,
        w_dispersion=0.45,
        w_revision=0.35,
        w_pct=0.20,
        pct_sign=-1.0,
        pct_cap=25.0,
        pct_transform="signed_log1p",
    ):
        self.revision_window = int(revision_window)
        self.revision_smooth_window = int(revision_smooth_window)
        self.span = int(smoothing_span)
        self.dispersion_window = int(dispersion_window)
        self.pct_smooth_window = int(pct_smooth_window)
        self.clip_dispersion = clip_dispersion
        self.w_dispersion = float(w_dispersion)
        self.w_revision = float(w_revision)
        self.w_pct = float(w_pct)
        self.pct_sign = float(pct_sign)
        self.pct_cap = pct_cap
        self.pct_transform = pct_transform

    def _normalize_weights(self) -> Tuple[float, float, float]:
        total = self.w_dispersion + self.w_revision + self.w_pct
        if total <= 0:
            raise ValueError("Sum of weights must be > 0.")
        return self.w_dispersion / total, self.w_revision / total, self.w_pct / total

    def _transform_pct(self, pct_df: pd.DataFrame) -> pd.DataFrame:
        pct = pct_df.copy()
        if self.pct_cap is not None:
            pct = pct.clip(lower=-self.pct_cap, upper=self.pct_cap)

        if self.pct_transform == "signed_log1p":
            return np.sign(pct) * np.log1p(np.abs(pct))
        if self.pct_transform == "tanh":
            scale = max(float(self.pct_cap or 50.0) / 3.0, 1e-6)
            return np.tanh(pct / scale)
        if self.pct_transform == "raw":
            return pct
        raise ValueError("pct_transform must be one of: 'signed_log1p', 'tanh', 'raw'.")

    def generate(
        self,
        closing_price: pd.DataFrame,
        eps_mean: pd.DataFrame,
        eps_std: pd.DataFrame,
        eps_pct: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        w_d, w_r, w_p = self._normalize_weights()
        print(
            f"  -> Generating EPS Forecast Signals (rev={self.revision_window}d, span={self.span}, weights={w_d:.2f}/{w_r:.2f}/{w_p:.2f})..."
        )



        dispersion = eps_std.div(closing_price.replace(0, np.nan))
        dispersion = dispersion.rolling(window=self.dispersion_window, min_periods=5).mean()
        if self.clip_dispersion is not None:
            dispersion = dispersion.clip(lower=-self.clip_dispersion, upper=self.clip_dispersion)

        revision = eps_mean.pct_change(periods=self.revision_window, fill_method=None).div(closing_price.replace(0, np.nan))
        revision = revision.rolling(window=self.revision_smooth_window, min_periods=5).mean()

        dispersion_z = robust_cross_sectional_norm(dispersion.replace([np.inf, -np.inf], np.nan).fillna(0.0))
        revision_z = robust_cross_sectional_norm(revision.replace([np.inf, -np.inf], np.nan).fillna(0.0))

        if eps_pct is not None and w_p > 0:
            eps_pct = eps_pct.rolling(window=self.pct_smooth_window, min_periods=5).mean()
            pct_transformed = self._transform_pct(eps_pct)
            pct_z = robust_cross_sectional_norm(pct_transformed.replace([np.inf, -np.inf], np.nan).fillna(0.0))
        else:
            common_index = closing_price.index.intersection(eps_mean.index).intersection(eps_std.index)
            common_cols = closing_price.columns.intersection(eps_mean.columns).intersection(eps_std.columns)
            pct_z = pd.DataFrame(0.0, index=common_index, columns=common_cols)

        alpha_raw = -(w_d * dispersion_z) + (w_r * revision_z) + (self.pct_sign * w_p * pct_z)

        smoothed = alpha_raw.ewm(span=self.span, min_periods=1).mean()
        eps_signal = robust_cross_sectional_norm(smoothed)

        return eps_signal
    

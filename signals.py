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



class ShortTermSignalGenerator:
    """Generates fast, mean-reversion signals based on short-term price action."""
    def __init__(self, reversal_window=5, smoothing_span=3):
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
        signal = -z_score
        
        # Smooth the raw signals if you aren't already
        sig_df = signal.ewm(span=self.span, min_periods=1).mean()

        # SWAP TO ROBUST NORMALIZATION
        final_z = robust_cross_sectional_norm(sig_df)
        
        return final_z



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
        self.freq = pca_update_freq # Re-run PCA and full HMM BIC check every X days
        self.n_states = initial_states
        self.max_states = max_states
        self.hmm_window = hmm_window # Rolling window for HMM to prevent infinite memory growth
        self.model = None

    def _calc_bic(self, model, X):
        """Calculates Bayesian Information Criterion for an HMM"""
        score = model.score(X) # Log-likelihood
        n_features = X.shape[1]
        n_states = model.n_components
        # Free parameters: Transitions + Means + Covariances (full)
        n_params = n_states*(n_states-1) + n_states*n_features + n_states*n_features*(n_features+1)/2
        return -2 * score + n_params * np.log(X.shape[0])

    def _fit_best_hmm(self, X, current_states):
        best_model = None
        
        # FIX: Changed to "diag" to prevent matrix singularity on orthogonal PCs
        model_curr = hmm.GaussianHMM(
            n_components=current_states, 
            covariance_type="diag", 
            n_iter=100, 
            random_state=42
        )
        model_curr.fit(X)
        bic_curr = self._calc_bic(model_curr, X)
        best_model = model_curr
        best_states = current_states
        
        if current_states < self.max_states:
            model_up = hmm.GaussianHMM(
                n_components=current_states + 1, 
                covariance_type="diag", 
                n_iter=100, 
                random_state=42
            )
            model_up.fit(X)
            bic_up = self._calc_bic(model_up, X)
            
            if bic_up < bic_curr:
                best_model = model_up
                best_states = current_states + 1
                
        return best_model, best_states

    def generate(self, returns):
        print(f"  -> Generating HMM-PCA Signals (Top {self.k} PCs, updating states every {self.freq} days)...")
        
        n_days, n_assets = returns.shape
        signal_matrix = np.zeros((n_days, n_assets))
        
        ret_vals = returns.fillna(0.0).values
        
        svd = TruncatedSVD(n_components=self.k, random_state=42)
        top_vecs = None
        pc_returns = np.zeros((n_days, self.k))
        
        # We need enough days to estimate K*K covariance matrices for the HMM states.
        # 60 days (roughly 3 months) is a safe minimum.
        burn_in = max(60, self.k * 3) 
        
        for i in range(1, n_days):
            
            # --- THE FIX: Skip until we have enough data ---
            if i < burn_in:
                continue # Signal remains 0.0 for the burn-in period
            
            # --- EXPENSIVE STEP: Periodic SVD & Full HMM BIC Check ---
            # It will naturally trigger on the first day after burn_in because top_vecs is None
            if i % self.freq == 0 or top_vecs is None:
                lookback_start = max(0, i - self.hmm_window)
                window_data = ret_vals[lookback_start:i]
                
                # SVD needs mean-centered data
                window_data_centered = window_data - np.mean(window_data, axis=0)
                svd.fit(window_data_centered)
                top_vecs = svd.components_.T 
                
                # 2. Re-project historical returns onto new PCs for the HMM
                pc_hist = window_data_centered @ top_vecs
                pc_returns[lookback_start:i] = pc_hist
                
                # 3. Refit HMM and check BIC
                self.model, self.n_states = self._fit_best_hmm(pc_hist, self.n_states)
                
            else:
                # --- CHEAP STEP: Daily updates ---
                today_centered = ret_vals[i] - np.mean(ret_vals[i])
                pc_returns[i] = today_centered @ top_vecs
                
                lookback_start = max(0, i - self.hmm_window)
                X_recent = pc_returns[lookback_start : i+1]
                
                self.model.init_params = '' 
                self.model.n_iter = 2
                self.model.fit(X_recent)
                
            # --- SIGNAL GENERATION (Mixture Sharpe) ---
            lookback_start = max(0, i - self.hmm_window)
                # --- HMM SANITIZER ---
            # If a state gets hollowed out during fitting, hmmlearn leaves a zero-sum row.
            # We must re-normalize the transition matrix to prevent predict_proba from crashing.
            if hasattr(self.model, 'transmat_'):
                row_sums = self.model.transmat_.sum(axis=1)
                for r_idx, r_sum in enumerate(row_sums):
                    if np.isclose(r_sum, 0.0) or np.isnan(r_sum):
                        # If the row collapsed, assign a uniform distribution
                        self.model.transmat_[r_idx, :] = 1.0 / self.model.n_components
                    else:
                        # Otherwise, force it to sum to exactly 1.0 to handle floating point errors
                        self.model.transmat_[r_idx, :] /= r_sum
            if hasattr(self.model, 'startprob_'):
                s_sum = np.sum(self.model.startprob_)
                if np.isclose(s_sum, 0.0) or np.isnan(s_sum):
                    # If corrupted, reset to a uniform distribution
                    self.model.startprob_ = np.ones(self.model.n_components) / self.model.n_components
                else:
                    # Force exact sum to 1.0
                    self.model.startprob_ /= s_sum
            filtered_probs = self.model.predict_proba(pc_returns[lookback_start : i+1])
            curr_state_prob = filtered_probs[-1] 
            
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
        
        # Assuming robust_cross_sectional_norm is in your scope
        final_z = robust_cross_sectional_norm(sig_df)
        
        return final_z



import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

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
        w_df = pd.DataFrame(daily_weights, index=hedged_returns.index, columns=signal_names)
        
        smoothed_coefs = w_df.rolling(self.lookback, min_periods=10).mean()
        
        scale_factor = smoothed_coefs.abs().mean(axis=1) + 1e-8
        scaled_coefs = smoothed_coefs.div(scale_factor, axis=0)
        
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
    
# class RollingLinearSignal(SignalGenerator):
#     """
#     A walk-forward rolling linear model (Ridge Regression).
    
#     Fixed to use Time-Series scaling instead of Cross-Sectional neutralization,
#     allowing the model to capture directional trends and fat-tailed momentum.
#     """
#     # Changed default fwd_days to 21 to match the refit window and avoid 1-day noise fitting
#     def __init__(self, train_window=252, refit_every=21, fwd_days=21, ridge_alpha=10.0):
#         self.train_window = train_window
#         self.refit_every = refit_every
#         self.fwd_days = fwd_days
#         self.ridge_alpha = ridge_alpha
#         self.history_coefs = []

#     def get_signals(self, hedged_returns, **kwargs):
#         log_returns = np.log1p(hedged_returns)
#         dates = log_returns.index
#         tickers = log_returns.columns
        
#         print("1. Computing base factors...")
#         mom = log_returns.rolling(252, min_periods=200).sum() - log_returns.rolling(21, min_periods=15).sum()
#         rev = -log_returns.rolling(5, min_periods=3).sum()
#         vol = -log_returns.rolling(60, min_periods=40).std(ddof=0)
        
#         print("2. Applying time-series scaling (Removed _cs_robust)...")
#         # Use a tiny epsilon to prevent division by zero on zero-volatility/flat days
#         eps = 1e-8
#         z_mom = mom / (mom.rolling(252, min_periods=100).std() + eps)
#         z_rev = rev / (rev.rolling(252, min_periods=100).std() + eps)
#         z_vol = vol / (vol.rolling(252, min_periods=100).std() + eps)
        
#         factors = {"mom": z_mom, "rev": z_rev, "vol": z_vol}
#         feat_names = list(factors.keys())
        
#         print(f"3. Building target variable (Predicting {self.fwd_days}-day returns)...")
#         # Target is raw forward returns. We do NOT neutralize the target.
#         fwd = log_returns.rolling(self.fwd_days).sum().shift(-self.fwd_days)
        
#         out_signal = pd.DataFrame(index=dates, columns=tickers, dtype=float)
        
#         # fit_intercept=True allows the model to handle the base market drift
#         model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        
#         start_idx = self.train_window + self.fwd_days
#         refits = list(range(start_idx, len(dates), self.refit_every))
        
#         print(f"4. Running walk-forward linear model ({len(refits)} refits)...")
        
#         is_fitted = False 
        
#         for rp in tqdm(refits, desc="Rolling Fit"):
#             tr1_idx = rp - self.fwd_days # Ensure NO lookahead bias
#             tr0_idx = max(0, tr1_idx - self.train_window)
#             train_dates = dates[tr0_idx:tr1_idx]
            
#             X_train, y_train = [], []
#             for t in train_dates:
#                 Xt = np.column_stack([factors[f].loc[t].values for f in feat_names])
#                 yt = fwd.loc[t].values
#                 valid = np.isfinite(Xt).all(axis=1) & np.isfinite(yt)
#                 if valid.sum() > 0:
#                     X_train.append(Xt[valid])
#                     y_train.append(yt[valid])
                
#             if X_train:
#                 X_train = np.vstack(X_train)
#                 y_train = np.concatenate(y_train)
                
#                 # Fit if we have enough data points
#                 if len(y_train) >= 100:
#                     model.fit(X_train, y_train)
#                     is_fitted = True
#                     self.history_coefs.append({
#                         "date": dates[rp],
#                         "mom_weight": model.coef_[0],
#                         "rev_weight": model.coef_[1],
#                         "vol_weight": model.coef_[2],
#                         "intercept": model.intercept_
#                     })
            
#             if not is_fitted:
#                 continue
                
#             nxt = min(len(dates), rp + self.refit_every)
#             block_dates = dates[rp:nxt]
            
#             # Predict
#             for t in block_dates:
#                 Xt = np.column_stack([factors[f].loc[t].values for f in feat_names])
#                 valid = np.isfinite(Xt).all(axis=1)
#                 if valid.sum() > 0:
#                     preds = model.predict(Xt[valid])
#                     out_signal.loc[t, tickers[valid]] = preds

#         # 5. Translate raw return predictions into cross-sectional Z-scores
#         # This gives the portfolio engine the relative sizing it expects (mean=0, std=1)
#         print("5. Formatting final signals for portfolio constructor...")
        
#         # Calculate daily cross-sectional mean and standard deviation
#         cs_mean = out_signal.mean(axis=1)
#         cs_std = out_signal.std(axis=1)
        
#         # Z-score the signals day-by-day (using a tiny epsilon to prevent division by zero)
#         final_signal = out_signal.sub(cs_mean, axis=0).div(cs_std + 1e-8, axis=0)
        
#         return final_signal

# class TimeSeriesMultiFactor(SignalGenerator):
    # """
    # Combines factors using historical time-series scaling rather than 
    # daily cross-sectional standardizing. This preserves outliers and 
    # overall market directionality.
    # """
    # def __init__(self, mom_weight=1.0, rev_weight=0.5, vol_weight=0.5):
    #     # Slightly down-weighting rev and vol to let momentum lead
    #     self.mom_weight = mom_weight
    #     self.rev_weight = rev_weight
    #     self.vol_weight = vol_weight

    # def get_signals(self, hedged_returns, **kwargs):
    #     log_returns = np.log1p(hedged_returns)
        
    #     # 1. Base factors
    #     mom = log_returns.rolling(252, min_periods=200).sum() - log_returns.rolling(21, min_periods=15).sum()
    #     rev = -log_returns.rolling(5, min_periods=3).sum()
    #     vol = -log_returns.rolling(60, min_periods=40).std(ddof=0)
        
    #     # 2. Time-Series Scaling (NOT Cross-Sectional)
    #     # We divide by the rolling 252-day standard deviation of the FACTOR ITSELF 
    #     # to normalize the scale, but without forcing the cross-section to sum to zero.
    #     mom_scaled = mom / mom.rolling(252, min_periods=100).std()
    #     rev_scaled = rev / rev.rolling(252, min_periods=100).std()
    #     vol_scaled = vol / vol.rolling(252, min_periods=100).std()
        
    #     # 3. Combine without clipping
    #     signal = (
    #         (mom_scaled * self.mom_weight) + 
    #         (rev_scaled * self.rev_weight) + 
    #         (vol_scaled * self.vol_weight)
    #     )
        
    #     # Return the raw combined signal. No _cs_robust!
    #     return signal


    # class DynamicSignalBlender:
#     """
#     Allocates weight using a convex combination. A fixed percentage is allocated 
#     to the PCA signal, and the remaining budget is dynamically split between 
#     Short and Long signals using a Composite Regime Model.
#     """
#     def __init__(self, fast_win=21, slow_win=252, trend_win=60, pca_weight=0.30):
#         self.fast_win = fast_win
#         self.slow_win = slow_win
#         self.trend_win = trend_win
#         self.pca_weight = pca_weight # Convex allocation to PCA signal

#     def blend(self, short_signals, long_signals, pca_signals, benchmark_returns):
#         print(f"  -> Blending signals (Convex Budget: PCA {self.pca_weight*100}%, Dynamic { (1-self.pca_weight)*100 }%)...")
        
#         baseline_lookback = 252 * 2 
        
#         # --- 1. Volatility Regime ---
#         bench_vol_fast = benchmark_returns.rolling(self.fast_win, min_periods=10).std()
#         bench_vol_slow = benchmark_returns.rolling(self.slow_win, min_periods=60).std()
#         vol_ratio = (bench_vol_fast / (bench_vol_slow + 1e-8)).fillna(1.0)
#         vol_z = (vol_ratio - vol_ratio.rolling(baseline_lookback, min_periods=126).mean()) / \
#                 (vol_ratio.rolling(baseline_lookback, min_periods=126).std() + 1e-8)
#         vol_score = 1 / (1 + np.exp(-vol_z.fillna(0)))

#         # --- 2. Trend Regime ---
#         trend_ret = benchmark_returns.rolling(self.trend_win).sum()
#         trend_vol = benchmark_returns.rolling(self.trend_win).std() * np.sqrt(self.trend_win)
#         trend_strength = (trend_ret / (trend_vol + 1e-8)).abs()
#         trend_z = (trend_strength - trend_strength.rolling(baseline_lookback, min_periods=126).mean()) / \
#                   (trend_strength.rolling(baseline_lookback, min_periods=126).std() + 1e-8)
#         trend_score = 1 / (1 + np.exp(-trend_z.fillna(0)))

#         # --- 3. Convex Weight Allocation ---
#         dynamic_budget = 1.0 - self.pca_weight
        
#         # Baseline 50/50 split of the dynamic budget, adjusted by regimes
#         raw_short_ratio = 0.5 + 0.5 * (vol_score - trend_score)
        
#         # Calculate actual convex weights
#         short_weight = raw_short_ratio.clip(lower=0.20, upper=0.80) * dynamic_budget
#         long_weight = dynamic_budget - short_weight
        
#         # --- 4. Apply and Re-Normalize ---
#         blended = (short_signals.mul(short_weight, axis=0) + 
#                    long_signals.mul(long_weight, axis=0) + 
#                    pca_signals.mul(self.pca_weight, axis=0))
#         cs_mean = blended.mean(axis=1)
#         cs_std = blended.std(axis=1)
#         final_signal = blended.sub(cs_mean, axis=0).div(cs_std + 1e-8, axis=0)
        
#         return final_signal

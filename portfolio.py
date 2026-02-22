import pandas as pd
import numpy as np
import scipy.stats as stats



class PortfolioConstructor:
    def __init__(self, target_ann_vol=500000, max_adv_pct=0.025, signal_threshold=0.75, 
                 hard_volume_limit=2000000, max_gross_exposure=10000000, corr_shrinkage=0.20):
        self.target_daily_vol = target_ann_vol / np.sqrt(252)
        self.max_adv_pct = max_adv_pct
        self.hard_volume_limit = hard_volume_limit
        self.signal_threshold = signal_threshold 
        self.max_gross_exposure = max_gross_exposure
        self.corr_shrinkage = corr_shrinkage # NEW: 20% shrinkage towards 0 correlation

    def _shrink_covariance(self, raw_cov_df):
        """
        Preserves the exact standard deviations (vols) but shrinks the 
        noisy off-diagonal correlations towards the identity matrix.
        """
        raw_cov = raw_cov_df.values
        vols = np.sqrt(np.diag(raw_cov))
        
        # Prevent divide-by-zero for any halted/flatlined stocks
        safe_vols = np.clip(vols, a_min=1e-6, a_max=None)
        outer_vols = np.outer(safe_vols, safe_vols)
        
        # Extract pure correlation matrix
        corr_matrix = raw_cov / outer_vols
        
        # Shrink correlations towards the Identity Matrix (0.0 correlation)
        identity_mat = np.eye(len(raw_cov))
        robust_corr = (1.0 - self.corr_shrinkage) * corr_matrix + (self.corr_shrinkage * identity_mat)
        
        # Reconstruct the covariance matrix with the original 60-day vols
        robust_cov = robust_corr * outer_vols
        return pd.DataFrame(robust_cov, index=raw_cov_df.index, columns=raw_cov_df.columns)

    def generate_target_positions(self, t, signals, cov_matrix, adv_60d, betas, benchmark_ticker):
        sig_t = signals.dropna()
        if len(sig_t) < 10:
            return pd.Series(0.0, index=signals.index)
            
        # --- 1. Soft Thresholding (The Significance Metric) ---
        active_signals = np.sign(sig_t) * np.maximum(0, sig_t.abs() - self.signal_threshold)
        active_assets = active_signals[active_signals != 0].index
        if len(active_assets) < 5:
             return pd.Series(0.0, index=signals.index)
             
        # --- 2. Risk Weighting (Using Robust Covariance) ---
        clean_cov = cov_matrix.loc[active_assets, active_assets].fillna(0.0)
        robust_cov = self._shrink_covariance(clean_cov)
        
        vols = np.sqrt(np.diag(robust_cov))
        vol_series = pd.Series(vols, index=active_assets).clip(lower=0.001)
        
        raw_weights = active_signals.loc[active_assets] / vol_series
        
        # --- 3. Separate and Normalize Long/Short Books ---
        longs = raw_weights[raw_weights > 0]
        shorts = raw_weights[raw_weights < 0]
        
        pos = pd.Series(0.0, index=signals.index)
        
        if len(longs) > 0 and len(shorts) > 0:
            # Force perfectly symmetric $ neutrality in abstract space
            pos.loc[longs.index] = longs / longs.sum()
            pos.loc[shorts.index] = shorts / abs(shorts.sum())
        else:
            return pos

        # --- 3.5 Abstract Beta Hedging ---
        assets_only = pos.index[pos != 0]
        abstract_beta_exposure = (pos[assets_only] * betas.loc[assets_only].fillna(1.0)).sum()
        pos[benchmark_ticker] = -abstract_beta_exposure

        # --- 4. Target Volatility Scaling & Gross Cap ---
        full_clean_cov = cov_matrix.loc[pos.index, pos.index].fillna(0.0)
        
        # Apply the same shrinkage to the full portfolio to ensure accurate target sizing
        full_robust_cov = self._shrink_covariance(full_clean_cov)
        port_vol = np.sqrt(pos.T @ full_robust_cov @ pos)
        
        if port_vol > 0:
            scalar = self.target_daily_vol / port_vol
            
            # Gross exposure check (excluding the benchmark from the cap)
            current_abstract_gross = pos[assets_only].abs().sum() 
            max_safe_scalar = self.max_gross_exposure / current_abstract_gross if current_abstract_gross > 0 else scalar
            
            final_scalar = min(scalar, max_safe_scalar)
            pos *= final_scalar
        else:
            return pd.Series(0.0, index=signals.index)

        # # --- 5. Liquidity Constraints (Strict Clipping) ---
        # max_pos = adv_60d.loc[assets_only].fillna(0.0) * self.max_adv_pct
        # max_pos = max_pos.clip(upper=self.hard_volume_limit)
        
        # pos.loc[assets_only] = pos.loc[assets_only].clip(lower=-max_pos, upper=max_pos)

        # # --- 6. Recalculate Final Benchmark Hedge ---
        # final_beta_exposure = (pos[assets_only] * betas.loc[assets_only].fillna(1.0)).sum()
        # pos[benchmark_ticker] = -final_beta_exposure

        # --- 5. Liquidity Constraints (Strict Clipping) ---
        max_pos = adv_60d.loc[assets_only].fillna(0.0) * self.max_adv_pct
        max_pos = max_pos.clip(upper=self.hard_volume_limit)
        
        pos.loc[assets_only] = pos.loc[assets_only].clip(lower=-max_pos, upper=max_pos)

        # --- NEW: 5.5 Re-force Dollar Neutrality Post-Clipping ---
        # Calculate the size of the long and short books AFTER clipping
        final_longs = pos.loc[assets_only][pos.loc[assets_only] > 0]
        final_shorts = pos.loc[assets_only][pos.loc[assets_only] < 0]
        
        sum_longs = final_longs.sum()
        sum_shorts = np.abs(final_shorts.sum())
        
        if sum_longs > 0 and sum_shorts > 0:
            # If longs are larger, the shorts were constrained. Scale longs down to match.
            if sum_longs > sum_shorts:
                pos.loc[final_longs.index] *= (sum_shorts / sum_longs)
            # If shorts are larger, the longs were constrained. Scale shorts down to match.
            elif sum_shorts > sum_longs:
                pos.loc[final_shorts.index] *= (sum_longs / sum_shorts)
        else:
            # Failsafe: if one entire side of the book was zeroed out by liquidity constraints
            pos.loc[assets_only] = 0.0

        # --- 6. Recalculate Final Benchmark Hedge ---
        final_beta_exposure = (pos[assets_only] * betas.loc[assets_only].fillna(1.0)).sum()
        pos[benchmark_ticker] = -final_beta_exposure

        return pos


class CurrencyNeutralPortfolioConstructor:
    def __init__(self, target_ann_vol, max_adv_pct, signal_threshold, hard_volume_limit, max_gross_exposure, currency_dict):
        self.target_ann_vol = target_ann_vol
        self.max_adv_pct = max_adv_pct
        self.signal_threshold = signal_threshold
        self.hard_volume_limit = hard_volume_limit
        self.max_gross_exposure = max_gross_exposure
        self.currency_dict = currency_dict # Passed from DataProcessor

    def generate_target_positions(self, t, signals, cov_matrix, adv_60d, betas, benchmark_ticker):
        # 1. Filter by signal strength
        active_assets = signals[signals.abs() > self.signal_threshold].index
        
        if len(active_assets) < 5:
            return pd.Series(0.0, index=signals.index)
            
        raw_weights = signals[active_assets].copy()
        
        # 2. CURRENCY NEUTRALIZATION STEP
        # Group the active assets by their currency and demean the weights
        # This guarantees net dollar exposure for EUR, GBP, SEK, etc., is exactly $0
        currency_groups = pd.Series([self.currency_dict.get(ric, 'EUR') for ric in active_assets], index=active_assets)
        
        for curr in currency_groups.unique():
            curr_assets = currency_groups[currency_groups == curr].index
            if len(curr_assets) > 1:
                # Demean weights within this specific currency
                raw_weights[curr_assets] -= raw_weights[curr_assets].mean()
            else:
                # If only 1 asset in a currency exists, we must drop it to maintain neutrality
                raw_weights[curr_assets] = 0.0 
                
        # 3. Volatility Scaling (Standard Risk Parity/Covariance Scaling)
        # Calculate expected portfolio variance
        w_array = raw_weights.values
        port_var = w_array.T @ cov_matrix.loc[active_assets, active_assets].values @ w_array
        
        if port_var <= 0:
            return pd.Series(0.0, index=signals.index)
            
        port_vol = np.sqrt(port_var * 252) # Annualized
        
        # Scale weights to hit the $500k USD volatility target
        vol_scalar = self.target_ann_vol / port_vol
        target_notionals = raw_weights * vol_scalar
        
        # 4. Apply ADV Limits and Hard Volume Limits (in USD)
        max_allowed_adv = adv_60d[active_assets] * self.max_adv_pct
        max_allowed = np.minimum(max_allowed_adv, self.hard_volume_limit)
        
        # Clip positions
        target_notionals = target_notionals.clip(lower=-max_allowed, upper=max_allowed)
        
        # 5. Apply Max Gross Exposure Limit ($10M USD)
        gross_exposure = target_notionals.abs().sum()
        if gross_exposure > self.max_gross_exposure:
            gross_scalar = self.max_gross_exposure / gross_exposure
            target_notionals *= gross_scalar
            
        # Re-index to full universe
        final_positions = target_notionals.reindex(signals.index).fillna(0.0)
        
        return final_positions
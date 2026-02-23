import pandas as pd
import numpy as np
import scipy.stats as stats

class PortfolioConstructor:
    def __init__(self, target_ann_vol=500000, max_adv_pct=0.025, signal_threshold=0.75, 
                 hard_volume_limit=2000000, max_gross_exposure=10000000, corr_shrinkage=0.20,
                 trade_speed=0.50, decay_allowance_ratio=0.50):
        self.target_daily_vol = target_ann_vol / np.sqrt(252)
        self.max_adv_pct = max_adv_pct
        self.hard_volume_limit = hard_volume_limit
        self.signal_threshold = signal_threshold 
        self.max_gross_exposure = max_gross_exposure
        self.corr_shrinkage = corr_shrinkage 
        
        # NEW: Turnover suppression controls
        self.trade_speed = trade_speed
        self.decay_allowance_ratio = decay_allowance_ratio

    def _shrink_covariance(self, raw_cov_df):
        """Preserves exact standard deviations but shrinks noisy correlations towards 0."""
        raw_cov = raw_cov_df.values
        vols = np.sqrt(np.diag(raw_cov))
        safe_vols = np.clip(vols, a_min=1e-6, a_max=None)
        outer_vols = np.outer(safe_vols, safe_vols)
        
        corr_matrix = raw_cov / outer_vols
        identity_mat = np.eye(len(raw_cov))
        robust_corr = (1.0 - self.corr_shrinkage) * corr_matrix + (self.corr_shrinkage * identity_mat)
        
        robust_cov = robust_corr * outer_vols
        return pd.DataFrame(robust_cov, index=raw_cov_df.index, columns=raw_cov_df.columns)

    def generate_target_positions(self, t, signals, cov_matrix, adv_60d, betas, benchmark_ticker, current_positions=None):
        sig_t = signals.dropna()
        if len(sig_t) < 10:
            return pd.Series(0.0, index=signals.index)
            
        # --- 1. SIGNAL HYSTERESIS (The Sticky Threshold) ---
        effective_threshold = pd.Series(self.signal_threshold, index=sig_t.index)
        
        if current_positions is not None:
            # Identify what we currently hold (excluding the benchmark hedge)
            currently_held = current_positions[current_positions != 0].drop(benchmark_ticker, errors='ignore').index
            # Lower the threshold for assets we already own to prevent premature liquidation
            decayed_thresh = self.signal_threshold * self.decay_allowance_ratio
            effective_threshold.loc[currently_held.intersection(sig_t.index)] = decayed_thresh
            
        active_signals = np.sign(sig_t) * np.maximum(0, sig_t.abs() - effective_threshold)
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
        full_robust_cov = self._shrink_covariance(full_clean_cov)
        port_vol = np.sqrt(pos.T @ full_robust_cov @ pos)
        
        if port_vol > 0:
            scalar = self.target_daily_vol / port_vol
            current_abstract_gross = pos[assets_only].abs().sum() 
            max_safe_scalar = self.max_gross_exposure / current_abstract_gross if current_abstract_gross > 0 else scalar
            
            final_scalar = min(scalar, max_safe_scalar)
            pos *= final_scalar
        else:
            return pd.Series(0.0, index=signals.index)

        # --- 5. Liquidity Constraints (Strict Clipping) ---
        max_pos = adv_60d.loc[assets_only].fillna(0.0) * self.max_adv_pct
        max_pos = max_pos.clip(upper=self.hard_volume_limit)
        
        pos.loc[assets_only] = pos.loc[assets_only].clip(lower=-max_pos, upper=max_pos)

        # --- 5.5 Re-force Dollar Neutrality Post-Clipping ---
        final_longs = pos.loc[assets_only][pos.loc[assets_only] > 0]
        final_shorts = pos.loc[assets_only][pos.loc[assets_only] < 0]
        
        sum_longs = final_longs.sum()
        sum_shorts = np.abs(final_shorts.sum())
        
        if sum_longs > 0 and sum_shorts > 0:
            if sum_longs > sum_shorts:
                pos.loc[final_longs.index] *= (sum_shorts / sum_longs)
            elif sum_shorts > sum_longs:
                pos.loc[final_shorts.index] *= (sum_longs / sum_shorts)
        else:
            pos.loc[assets_only] = 0.0

        # --- 6. Recalculate Final Benchmark Hedge ---
        assets_only = pos.index[pos.index != benchmark_ticker]
        
        if current_positions is not None:
            aligned_current = current_positions.reindex(pos.index).fillna(0.0)
            
            # Fix: Base the tolerance on the LARGER of the current or target position
            # This prevents the buffer from shrinking to 0 during position exits
            max_pos_size = np.maximum(pos.loc[assets_only].abs(), aligned_current.loc[assets_only].abs())
            
            # Fix: Add a $2,500 absolute minimum buffer to ignore "dust" trades
            drift_tolerance = (max_pos_size * 0.15) + 2500 
            
            weight_diff = (pos.loc[assets_only] - aligned_current.loc[assets_only]).abs()
            
            inside_buffer = weight_diff <= drift_tolerance
            pos.loc[assets_only[inside_buffer]] = aligned_current.loc[assets_only[inside_buffer]]

        # --- 7. LINEAR TRADE DAMPENING (Turnover Reduction) ---
        if current_positions is not None:
            # Dampen ASSETS ONLY. Do not dampen the benchmark hedge!
            pos.loc[assets_only] = (aligned_current.loc[assets_only] * (1.0 - self.trade_speed)) + (pos.loc[assets_only] * self.trade_speed)

        # --- 8. RECALCULATE FINAL BENCHMARK HEDGE ---
        # The hedge must be based on the ACTUAL damped positions we are taking today
        final_beta_exposure = (pos.loc[assets_only] * betas.loc[assets_only].fillna(1.0)).sum()
        pos[benchmark_ticker] = -final_beta_exposure

        return pos



class CurrencyNeutralPortfolioConstructor:
    def __init__(self, target_ann_vol, max_adv_pct, signal_threshold, hard_volume_limit, max_gross_exposure, currency_dict,
                 trade_speed=0.50, decay_allowance_ratio=0.50, corr_shrinkage=0.20):
        self.target_ann_vol = target_ann_vol
        self.max_adv_pct = max_adv_pct
        self.signal_threshold = signal_threshold
        self.hard_volume_limit = hard_volume_limit
        self.max_gross_exposure = max_gross_exposure
        self.currency_dict = currency_dict 
        
        # Turnover suppression controls
        self.trade_speed = trade_speed
        self.decay_allowance_ratio = decay_allowance_ratio
        
        # NEW: Covariance shrinkage parameter
        self.corr_shrinkage = corr_shrinkage

    def _shrink_covariance(self, raw_cov_df):
        """Preserves exact standard deviations but shrinks noisy correlations towards 0."""
        raw_cov = raw_cov_df.values
        vols = np.sqrt(np.diag(raw_cov))
        safe_vols = np.clip(vols, a_min=1e-6, a_max=None)
        outer_vols = np.outer(safe_vols, safe_vols)
        
        corr_matrix = raw_cov / outer_vols
        identity_mat = np.eye(len(raw_cov))
        robust_corr = (1.0 - self.corr_shrinkage) * corr_matrix + (self.corr_shrinkage * identity_mat)
        
        robust_cov = robust_corr * outer_vols
        return pd.DataFrame(robust_cov, index=raw_cov_df.index, columns=raw_cov_df.columns)

    def generate_target_positions(self, t, signals, cov_matrix, adv_60d, betas, benchmark_ticker, current_positions=None):
        sig_t = signals.dropna()
        if len(sig_t) < 10:
            return pd.Series(0.0, index=signals.index)
            
        # --- 1. SIGNAL HYSTERESIS ---
        effective_threshold = pd.Series(self.signal_threshold, index=sig_t.index)
        
        if current_positions is not None:
            currently_held = current_positions[current_positions != 0].index
            decayed_thresh = self.signal_threshold * self.decay_allowance_ratio
            effective_threshold.loc[currently_held.intersection(sig_t.index)] = decayed_thresh
            
        active_signals = np.sign(sig_t) * np.maximum(0, sig_t.abs() - effective_threshold)
        active_assets = active_signals[active_signals != 0].index
        
        if len(active_assets) < 5:
            return pd.Series(0.0, index=signals.index)
            
        raw_weights = active_signals[active_assets].copy()
        
        # --- 2. CURRENCY NEUTRALIZATION STEP ---
        currency_groups = pd.Series([self.currency_dict.get(ric, 'EUR') for ric in active_assets], index=active_assets)
        
        for curr in currency_groups.unique():
            curr_assets = currency_groups[currency_groups == curr].index
            if len(curr_assets) > 1:
                raw_weights[curr_assets] -= raw_weights[curr_assets].mean()
            else:
                raw_weights[curr_assets] = 0.0 
                
        # --- 3. Volatility Scaling (Upgraded with Shrinkage) ---
        # Apply shrinkage to the active covariance matrix to prevent optimizer from leaning on noise
        clean_cov = cov_matrix.loc[active_assets, active_assets].fillna(0.0)
        robust_cov = self._shrink_covariance(clean_cov)
        
        w_array = raw_weights.values
        port_var = w_array.T @ robust_cov.values @ w_array
        
        if port_var <= 0:
            return pd.Series(0.0, index=signals.index)
            
        port_vol = np.sqrt(port_var * 252) 
        vol_scalar = self.target_ann_vol / port_vol
        target_notionals = raw_weights * vol_scalar
        
        # --- 4. Apply ADV Limits and Hard Volume Limits ---
        max_allowed_adv = adv_60d[active_assets] * self.max_adv_pct
        max_allowed = np.minimum(max_allowed_adv, self.hard_volume_limit)
        target_notionals = target_notionals.clip(lower=-max_allowed, upper=max_allowed)
        
        # --- 5. Apply Max Gross Exposure Limit ---
        gross_exposure = target_notionals.abs().sum()
        if gross_exposure > self.max_gross_exposure:
            gross_scalar = self.max_gross_exposure / gross_exposure
            target_notionals *= gross_scalar
            
        final_positions = target_notionals.reindex(signals.index).fillna(0.0)

        # --- FLAT COST DEADBAND (NO-TRADE ZONE) ---
        if current_positions is not None:
            aligned_current = current_positions.reindex(final_positions.index).fillna(0.0)

            # Base the tolerance on the larger of the current or target position
            max_pos_size = np.maximum(final_positions.abs(), aligned_current.abs())
            
            # Allow 15% drift + $2,500 absolute buffer to ignore tiny trades
            drift_tolerance = (max_pos_size * 0.15) + 2500

            weight_diff = (final_positions - aligned_current).abs()

            inside_buffer = weight_diff <= drift_tolerance
            final_positions[inside_buffer] = aligned_current[inside_buffer]

        # --- 7. LINEAR TRADE DAMPENING ---
        if current_positions is not None:
            final_positions = (aligned_current * (1.0 - self.trade_speed)) + (final_positions * self.trade_speed)
        
        return final_positions
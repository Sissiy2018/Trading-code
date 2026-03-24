import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import os

try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '3')
except NameError:
    pass

from data_loader import PipelineDataLoader, EuropeanDataLoader
from signals import (ShortTermSignalGenerator, LongTermSignalGenerator, PCASignalGenerator,
                     RobustRegressionBlender, RegimePCAHMMGenerator, VolumeConvictionGenerator,
                     DefensiveSignalGenerator, EPSRevisionGenerator, robust_cross_sectional_norm)
import glob
from portfolio import PortfolioConstructor, CurrencyNeutralPortfolioConstructor, USPortfolioConstructor
import config_signals
import config_daily as config
from backtester import Backtester
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def run_regional_pipeline(region='US'):
    print(f"\n========================================")
    print(f"       STARTING {region} PIPELINE")
    print(f"========================================")
    
    # 1. LOAD DATA & TARGET CONFIG
    if region == 'US':
        loader = PipelineDataLoader(benchmark_ticker='SPX')
        benchmark = 'SPX'
        cfg = config_signals.US
    else:
        loader = EuropeanDataLoader(benchmark_ticker='SX5E')
        benchmark = 'SX5E'
        cfg = config_signals.EU
        
    data = loader.fetch_all()

    def get_sign(ic):
        return -1 if ic < 0 else 1

    print(f"[{region}] Generating Alphas...")

    if region == 'US':
        # =================================================================
        # US STRATEGY: 3 signals (long, eps_rev, volume), equal-weight, no blender
        # =================================================================
        # Load EPS data (US only)
        eps_files = sorted([f for f in glob.glob(os.path.join(config.EPS_DIR, '*.csv')) if 'ADVfiltered' not in f])
        print(f"[US] Loading {len(eps_files)} EPS estimate files...")
        eps_raw = pd.concat([pd.read_csv(f) for f in eps_files], ignore_index=True)
        eps_raw['Date'] = pd.to_datetime(eps_raw['Date'])
        eps_raw = eps_raw.sort_values(['Date', 'RIC']).drop_duplicates(subset=['Date', 'RIC'], keep='last')
        eps_pivot = eps_raw.pivot_table(index='Date', columns='RIC', values='Earnings Per Share - Mean', aggfunc='last').sort_index()

        ic_long   = cfg['long_term'].get('IC', 0.0)
        ic_eps    = cfg['eps_revision'].get('IC', 0.0)
        ic_volume = cfg['volume'].get('IC', 0.0)

        sig_long = LongTermSignalGenerator(
            momentum_window=cfg['long_term']['momentum_window'],
            skip_recent=cfg['long_term']['skip_recent'],
            smoothing_span=cfg['long_term']['smoothing_span'],
            value_tilt_strength=cfg['long_term']['value_tilt_strength']
        ).generate(data.hedged_returns, data.earnings_yield, data.sectors) * get_sign(ic_long)

        sig_eps = EPSRevisionGenerator(
            revision_window=cfg['eps_revision']['revision_window'],
            smoothing_span=cfg['eps_revision']['smoothing_span']
        ).generate(data.hedged_returns, eps_pivot) * get_sign(ic_eps)

        sig_volume = VolumeConvictionGenerator(
            volume_window=cfg['volume']['volume_window'],
            return_window=cfg['volume']['return_window'],
            smoothing_span=cfg['volume']['smoothing_span']
        ).generate(data.hedged_returns, data.volume_usd) * get_sign(ic_volume)

        # Equal-weight blend: momentum + EPS revision + volume conviction
        final_signals = robust_cross_sectional_norm(sig_long + sig_eps + sig_volume)
        historical_weights = pd.DataFrame(
            {"long": 1/3, "eps_rev": 1/3, "volume": 1/3},
            index=data.hedged_returns.index
        )
        avg_horizon = (cfg['long_term']['horizon'] + cfg['eps_revision']['horizon'] + cfg['volume']['horizon']) / 3
        print(f"[US] Equal-weight: long + eps_rev + volume (long-only + index hedge)")

    else:
        # =================================================================
        # EU STRATEGY: Original 6 signals + Huber blender
        # =================================================================
        ic_short  = cfg['short_term'].get('IC', 0.0)
        ic_long   = cfg['long_term'].get('IC', 0.0)
        ic_pca    = cfg['pca'].get('IC', 0.0)
        ic_def    = cfg['defensive'].get('IC', 0.0)
        ic_regime = cfg['hmm'].get('IC', 0.0)
        ic_volume = 0

        signal_dict = {
            "short": ShortTermSignalGenerator(
                reversal_window=cfg['short_term']['reversal_window'],
                smoothing_span=cfg['short_term']['smoothing_span']
            ).generate(data.hedged_returns) * get_sign(ic_short),
            "long": LongTermSignalGenerator(
                momentum_window=cfg['long_term']['momentum_window'],
                skip_recent=cfg['long_term']['skip_recent'],
                smoothing_span=cfg['long_term']['smoothing_span'],
                value_tilt_strength=cfg['long_term']['value_tilt_strength']
            ).generate(data.hedged_returns, data.earnings_yield, data.sectors) * get_sign(ic_long),
            "pca": PCASignalGenerator(
                n_components=cfg['pca']['n_components'],
                cov_window=cfg['pca']['cov_window'],
                mom_window=cfg['pca']['mom_window'],
                rev_window=cfg['pca']['rev_window'],
                span=cfg['pca']['span'], pca_update_freq=21
            ).generate(data.hedged_returns) * get_sign(ic_pca),
            "volume": VolumeConvictionGenerator(
                volume_window=cfg['volume']['volume_window'],
                return_window=cfg['volume']['return_window'],
                smoothing_span=cfg['volume']['smoothing_span']
            ).generate(data.hedged_returns, data.volume_usd),
            "regime": RegimePCAHMMGenerator(
                n_components=cfg['hmm']['n_components'],
                pca_update_freq=cfg['hmm']['pca_update_freq'],
                max_states=cfg['hmm']['max_states'],
                hmm_window=cfg['hmm']['hmm_window']
            ).generate(data.hedged_returns) * get_sign(ic_regime),
            "defensive": DefensiveSignalGenerator(
                drift_window=cfg['defensive']['drift_window'],
                vol_window=cfg['defensive']['vol_window'],
                smoothing_span=cfg['defensive']['smoothing_span']
            ).generate(data.hedged_returns, data.betas) * get_sign(ic_def)
        }

        abs_ics = [abs(ic_short), abs(ic_long), abs(ic_pca), abs(ic_volume), abs(ic_regime), abs(ic_def)]
        total_abs_ic = sum(abs_ics)
        if total_abs_ic == 0:
            raise ValueError(f"[EU] All signals have zero IC!")
        priors = [ic / total_abs_ic for ic in abs_ics]

        print(f"[EU] Flipped? Short: {'Yes' if ic_short < 0 else 'No'}, Def: {'Yes' if ic_def < 0 else 'No'}, HMM: {'Yes' if ic_regime < 0 else 'No'}")
        print(f"[EU] Priors: Short:{priors[0]:.2f}, Long:{priors[1]:.2f}, PCA:{priors[2]:.2f}, Vol:{priors[3]:.2f}, HMM:{priors[4]:.2f}, Def:{priors[5]:.2f}")

        blender = RobustRegressionBlender(lookback=252, temperature=1.5)
        final_signals = blender.blend(signal_dict, data.hedged_returns, prior_weights=priors)
        historical_weights = blender.historical_weights
        avg_horizon = (
            (cfg['short_term']['horizon'] * priors[0]) +
            (cfg['long_term']['horizon'] * priors[1]) +
            (cfg['pca']['horizon'] * priors[2]) +
            (cfg['volume']['horizon'] * priors[3]) +
            (cfg['hmm']['horizon'] * priors[4]) +
            (cfg['defensive']['horizon'] * priors[5])
        )

    dynamic_trade_speed = max(0.05, min(1.0, 2.0 / (avg_horizon + 1)))
    print(f"[{region}] Blended Horizon: {avg_horizon:.1f} days -> Dynamic Trade Speed: {dynamic_trade_speed:.3f}")

    print(f"[{region}] Constructing Portfolio...")
    adv_60d = data.volume_usd.rolling(window=60, min_periods=10).mean()

    all_target_positions = pd.DataFrame(0.0, index=data.price_ret.index, columns=data.price_ret.columns)
    current_positions = pd.Series(0.0, index=data.price_ret.columns)
    warmup = 252 + 60

    if region == 'US':
        # =============================================================
        # US: LONG-ONLY + INDEX HEDGE (no individual stock shorts)
        # =============================================================
        portfolio_constructor = USPortfolioConstructor(
            target_ann_vol=config.PARAMS['TARGET_ANN_VOL'],
            max_adv_pct=config.PARAMS['MAX_ADV_PCT'],
            signal_threshold=0.55,
            hard_volume_limit=2000000,
            max_gross_exposure=10000000,
            trade_speed=0.15
        )

        for i, t in enumerate(data.price_ret.index):
            if i < warmup:
                continue
            if i % config.PARAMS['REBALANCE_FREQ_DAYS'] == 0:
                sig_t = final_signals.loc[t]
                if sig_t.isna().all():
                    all_target_positions.loc[t] = current_positions
                    continue

                active_assets = sig_t[sig_t > portfolio_constructor.signal_threshold].drop(benchmark, errors='ignore').index
                if len(active_assets) < 5:
                    all_target_positions.loc[t] = current_positions
                    continue

                cov_matrix_small = data.tot_ret_clean[active_assets].loc[:t].iloc[-60:].cov()
                cov_matrix = cov_matrix_small.reindex(index=data.price_ret.columns, columns=data.price_ret.columns, fill_value=0.0)

                current_positions = portfolio_constructor.generate_target_positions(
                    t=t, signals=sig_t, cov_matrix=cov_matrix,
                    adv_60d=adv_60d.loc[t], betas=data.betas.loc[t], benchmark_ticker=benchmark,
                    current_positions=current_positions, sectors=data.sectors
                )
            else:
                daily_total_ret = data.price_ret.loc[t].fillna(0) + data.div_ret.loc[t].fillna(0)
                current_positions = current_positions * (1 + daily_total_ret)

            all_target_positions.loc[t] = current_positions

    else:
        # =============================================================
        # EU: ORIGINAL LONG-SHORT WITH CURRENCY-NEUTRAL CONSTRUCTOR
        # =============================================================
        curr_dict = getattr(data, 'currency_dict', None)
        portfolio_constructor = CurrencyNeutralPortfolioConstructor(
            target_ann_vol=config.PARAMS['TARGET_ANN_VOL'],
            max_adv_pct=config.PARAMS['MAX_ADV_PCT'],
            signal_threshold=0.75,
            hard_volume_limit=2000000,
            max_gross_exposure=10000000,
            currency_dict=curr_dict,
            trade_speed=dynamic_trade_speed
        )

        for i, t in enumerate(data.price_ret.index):
            if i < warmup:
                continue
            if i % config.PARAMS['REBALANCE_FREQ_DAYS'] == 0:
                sig_t = final_signals.loc[t]
                if sig_t.isna().all():
                    all_target_positions.loc[t] = current_positions
                    continue

                active_assets = sig_t[sig_t.abs() > portfolio_constructor.signal_threshold].index
                if len(active_assets) < 5:
                    all_target_positions.loc[t] = current_positions
                    continue

                cov_matrix_small = data.tot_ret_clean[active_assets].loc[:t].iloc[-60:].cov()
                cov_matrix = cov_matrix_small.reindex(index=data.price_ret.columns, columns=data.price_ret.columns, fill_value=0.0)

                current_positions = portfolio_constructor.generate_target_positions(
                    t=t, signals=sig_t, cov_matrix=cov_matrix,
                    adv_60d=adv_60d.loc[t], betas=data.betas.loc[t], benchmark_ticker=benchmark,
                    current_positions=current_positions
                )
            else:
                daily_total_ret = data.price_ret.loc[t].fillna(0) + data.div_ret.loc[t].fillna(0)
                current_positions = current_positions * (1 + daily_total_ret)

            all_target_positions.loc[t] = current_positions

    return data, all_target_positions, loader, historical_weights


# =================================================================================
# 1. RUN INDIVIDUAL PIPELINES
# =================================================================================
us_data, us_positions, us_loader, us_weights = run_regional_pipeline('US')
eu_data, eu_positions, eu_loader, eu_weights = run_regional_pipeline('EU')
# =================================================================================
# 2. GLOBAL COMBINATION & DYNAMIC SOFTMAX WEIGHTING
# =================================================================================
print("\n========================================")
print("     PHASE 3: GLOBAL DYNAMIC COMBINER")
print("========================================")

# Align dates to the common trading days across both regions
common_dates = us_positions.index.intersection(eu_positions.index)
us_pos_aligned = us_positions.loc[common_dates]
eu_pos_aligned = eu_positions.loc[common_dates]

us_tot_ret = (us_data.price_ret.loc[common_dates].fillna(0) + us_data.div_ret.loc[common_dates].fillna(0))
eu_tot_ret = (eu_data.price_ret.loc[common_dates].fillna(0) + eu_data.div_ret.loc[common_dates].fillna(0))

# A. Calculate simulated daily PnL for the un-combined books
us_pnl = (us_pos_aligned.shift(1) * us_tot_ret).sum(axis=1)
eu_pnl = (eu_pos_aligned.shift(1) * eu_tot_ret).sum(axis=1)

# B. Calculate 252-day Rolling Sharpe for Softmax
roll_window = 252
us_roll_sharpe = (us_pnl.rolling(roll_window).mean() / (us_pnl.rolling(roll_window).std() + 1e-8)).fillna(0)
eu_roll_sharpe = (eu_pnl.rolling(roll_window).mean() / (eu_pnl.rolling(roll_window).std() + 1e-8)).fillna(0)

# C. Softmax Regression for smooth capital allocation
temperature = 0.1
exp_us = np.exp(us_roll_sharpe / temperature)
exp_eu = np.exp(eu_roll_sharpe / temperature)

dynamic_weight_us = exp_us / (exp_us + exp_eu)
dynamic_weight_eu = exp_eu / (exp_us + exp_eu)

# --- NEW: BAYESIAN PRIOR BLENDING ---
# 1. Define your structural base weights
prior_us = 0.3
prior_eu = 0.7

# 2. Define how strongly you trust the prior vs. the dynamic momentum (0.0 to 1.0)
#    0.0 = Fully dynamic (ignores prior), 1.0 = Fully static (pegs to prior)
prior_confidence = 0.4
raw_weight_us = (prior_us * prior_confidence) + (dynamic_weight_us * (1 - prior_confidence))
raw_weight_eu = (prior_eu * prior_confidence) + (dynamic_weight_eu * (1 - prior_confidence))

# D. Apply 60-day EMA to make the weights incredibly smooth and stable
# Note: we fill NaNs with the prior so the warmup period defaults to your baseline
weight_us = raw_weight_us.ewm(span=60).mean().fillna(prior_us)
weight_eu = raw_weight_eu.ewm(span=60).mean().fillna(prior_eu)

print(f"  Applying Dynamic Weights (Prior US/EU: {prior_us}/{prior_eu}, Confidence: {prior_confidence})...")
us_pos_weighted = us_pos_aligned.multiply(weight_us, axis=0)
eu_pos_weighted = eu_pos_aligned.multiply(weight_eu, axis=0)

# =================================================================================
# 3. GLOBAL VOLATILITY SCALING
# =================================================================================
print("  Applying Global Volatility Scaling (Target: 500k USD)...")
# Calculate the combined daily PnL of the weighted portfolio
combined_weighted_pnl = (us_pos_weighted.shift(1) * us_tot_ret).sum(axis=1) + (eu_pos_weighted.shift(1) * eu_tot_ret).sum(axis=1)

# Calculate rolling 60-day realized volatility of the combined book
rolling_global_vol = combined_weighted_pnl.rolling(60, min_periods=20).std() * np.sqrt(252)

# Calculate the scale factor required to bump the diversified book back to 500k
target_ann_vol = config.PARAMS['TARGET_ANN_VOL']
vol_scale_factor = (target_ann_vol / (rolling_global_vol + 1e-6)).clip(0.5, 3.0) # Cap leverage at 3x
vol_scale_factor = vol_scale_factor.ewm(span=10).mean().fillna(1.0) # Smooth the scaler slightly

us_global_final = us_pos_weighted.multiply(vol_scale_factor, axis=0)
eu_global_final = eu_pos_weighted.multiply(vol_scale_factor, axis=0)

# Combine into one massive global position matrix
global_positions = pd.concat([us_global_final, eu_global_final], axis=1)
global_positions=global_positions.iloc[:-1]
# =================================================================================
# 4. LIVE EXECUTION EXPORT (US AND EU)
# =================================================================================
# print("\n========================================")
# print("         PHASE 4: LIVE EXECUTION EXPORT")
# print("========================================")
# last_date = common_dates[-1]
# print(f"Valid global signals generated for date: {last_date.date()}")
# print(f"Current Global Allocation -> US: {weight_us.iloc[-1]:.1%}, EU: {weight_eu.iloc[-1]:.1%} (Scale Factor: {vol_scale_factor.iloc[-1]:.2f}x)")

# # --- US EXPORT ---
# us_active = us_global_final.iloc[-1]
# # us_active = us_active[us_active != 0].copy()
# us_exec = pd.DataFrame({
#     'internal_code': us_active.index,
#     'currency': 'USD',
#     'target_notional': us_active.values.round(2)
# })
# us_exec.to_csv('target_notionals_us_t_plus_1.csv', index=False)
# print(f"Saved {len(us_exec)} US target positions.")

# # --- EU EXPORT (With FX Translation) ---
# eu_active = eu_global_final.iloc[-1]
# # eu_active = eu_active[eu_active != 0].copy()

# fx_multipliers = eu_loader.processor.process_fx(config.EU_FX_FILES)
# latest_fx = fx_multipliers.reindex(eu_data.price_ret.index).ffill().loc[last_date]

# eu_exec = pd.DataFrame({
#     'internal_code': eu_active.index,
#     'currency': [eu_data.currency_dict.get(ric, 'EUR').upper() for ric in eu_active.index], 
#     'target_notional_usd': eu_active.values
# })

# def get_local_notional_fixed(row):
#     raw_curr = row['currency']
#     fx_col = f"{raw_curr}="
    
#     # Check if we have the specific rate, otherwise fallback to EUR
#     rate = latest_fx.get(fx_col, latest_fx.get('EUR=', 1.0))
#     local_val = row['target_notional_usd'] / rate
    
#     # PER USER INSTRUCTIONS: GBp is already correct as GBP value, so no /100 needed.
#     return local_val

# eu_exec['target_notional'] = eu_exec.apply(get_local_notional_fixed, axis=1).round(2)
# eu_exec['currency'] = eu_exec['currency'].apply(lambda x: 'GBP' if x == 'GBP' else x) # Ensure clean label
# eu_exec = eu_exec[['internal_code', 'currency', 'target_notional']]
# eu_exec.to_csv('target_notionals_eu_t_plus_1.csv', index=False)
# print(f"Saved {len(eu_exec)} EU target positions.")

# =================================================================================
# 5. GLOBAL BACKTESTING & ADVANCED REPORTING
# =================================================================================
print("\n========================================")
print("       PHASE 5: GLOBAL BACKTEST")
print("========================================")

# Combine returns and div returns globally 
global_price_ret = pd.concat([us_data.price_ret.loc[common_dates], eu_data.price_ret.loc[common_dates]], axis=1)
global_div_ret = pd.concat([us_data.div_ret.loc[common_dates], eu_data.div_ret.loc[common_dates]], axis=1)

# Run Backtest
global_backtester = Backtester(benchmark_ticker='SPX', tcost_bps=config.PARAMS['TCOST_BPS'], div_tax_rate=config.PARAMS['DIV_TAX'])
results = global_backtester.run(global_price_ret, global_div_ret, global_positions)

# --- ADVANCED METRICS CALCULATION ---
net_pnl = results['Net PnL'] # Daily dollar PnL
cum_pnl = results['Cumulative PnL']
# Return Metrics
annualized_pnl = net_pnl.mean() * 252
annualized_vol = net_pnl.std() * np.sqrt(252)
sharpe = annualized_pnl / annualized_vol if annualized_vol > 0 else 0

# Drawdown Calculation
rolling_max = cum_pnl.cummax()
drawdown = cum_pnl - rolling_max
max_drawdown = drawdown.min()

# Benchmark Correlation (Are we actually market neutral?)
# Re-align benchmark series just in case of missing days
us_bench = us_data.benchmark_series.pct_change().reindex(common_dates).ffill()
eu_bench = eu_data.benchmark_series.pct_change().reindex(common_dates).ffill()
corr_spx = net_pnl.corr(us_bench)
corr_sx5e = net_pnl.corr(eu_bench)

# Position & Execution Metrics
gross_exposure = global_positions.abs().sum(axis=1)
net_exposure = global_positions.sum(axis=1)
avg_gross = gross_exposure.mean()
avg_net = net_exposure.mean()

# Turnover (Annualized total dollars traded)
daily_turnover = global_positions.diff().abs().sum(axis=1)
annual_turnover = daily_turnover.mean() * 252
turnover_pct = (annual_turnover / avg_gross) * 100 if avg_gross > 0 else 0

total_tcosts = results['T-Costs'].sum()
total_financing = results['Financing'].sum()

# --- CONSOLE REPORT ---
print(f"\n[ RETURN METRICS ]")
print(f"  Cumulative PnL:       ${cum_pnl.iloc[-2]:,.2f}")
print(f"  Annualized PnL:       ${annualized_pnl:,.2f}")
print(f"  Annualized Vol:       ${annualized_vol:,.2f} (Target: $500,000)")
print(f"  Sharpe Ratio:         {sharpe:.3f}")
print(f"  Max Drawdown:         ${max_drawdown:,.2f}")

print(f"\n[ RISK & CORRELATION ]")
print(f"  Correlation vs SPX:   {corr_spx:.3f}")
print(f"  Correlation vs SX5E:  {corr_sx5e:.3f}")
print(f"  Avg Gross Exposure:   ${avg_gross:,.2f}")
print(f"  Avg Net Exposure:     ${avg_net:,.2f}")
print(f"  Final US Weight:      {weight_us.iloc[-1]:.1%}")
print(f"  Final EU Weight:      {weight_eu.iloc[-1]:.1%}")
print(f"  Hit Rate:            { ((net_pnl > 0).sum() / (net_pnl != 0).sum() * 100):.2f}%") 

print(f"\n[ FRICTION & EXECUTION ]")
print(f"  Annualized Turnover:  ${annual_turnover:,.2f} ({turnover_pct:.1f}% of Gross)")
print(f"  Total T-Costs:        ${total_tcosts:,.2f}")
print(f"  Total Financing:      ${total_financing:,.2f}")
print(f"  Total Friction Drag:  ${(total_tcosts + total_financing):,.2f}")

# --- PLOTTING ---
# ---> CHANGED: Increased to 6 subplots and taller figsize
fig, axes = plt.subplots(6, 1, figsize=(14, 24), sharex=True)

# 1. Cumulative PnL & Drawdown
cum_pnl.plot(ax=axes[0], color='forestgreen', lw=2, label='Cumulative PnL')
axes[0].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
axes[0].set_title('Global Portfolio: Cumulative Net PnL vs Drawdown (USD)')
axes[0].axhline(0, color='black', ls='--', alpha=0.4)
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# 2. Dynamic Weights Plot
weight_df = pd.DataFrame({'US Allocation': weight_us, 'EU Allocation': weight_eu})
weight_df.plot(ax=axes[1], kind='area', stacked=True, color=['steelblue', 'darkorange'], alpha=0.6)
axes[1].set_title('Dynamic Regional Allocation (Softmax + Prior Blending)')
axes[1].set_ylabel('Capital Weight')
axes[1].set_ylim(0, 1)

# 3. Scale Factor
vol_scale_factor.plot(ax=axes[2], color='purple', lw=2)
axes[2].set_title('Global Volatility Diversification Multiplier')
axes[2].grid(True, alpha=0.3)

# 4. Rolling Sharpe
rolling_sharpe = np.sqrt(252) * net_pnl.rolling(window=252).mean() / net_pnl.rolling(window=252).std()
rolling_sharpe.plot(ax=axes[3], color='orange', lw=2)
axes[3].set_title('Rolling 252-Day Sharpe Ratio')
axes[3].axhline(0, color='black', ls='--', alpha=0.4)
axes[3].axhline(1, color='green', ls='--', alpha=0.4)
axes[3].grid(True, alpha=0.3)

# ---> NEW: 5. US Signal Weights
# Align the weights to the common_dates to keep the x-axis consistent
us_weights_aligned = us_weights.loc[common_dates].ffill()
us_weights_aligned.plot(ax=axes[4], kind='area', stacked=True, colormap='tab10', alpha=0.7)
axes[4].set_title('US Internal Signal Allocation (Softmax Weights)')
axes[4].set_ylabel('Signal Weight')
axes[4].set_ylim(0, 1)
axes[4].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

# ---> NEW: 6. EU Signal Weights
eu_weights_aligned = eu_weights.loc[common_dates].ffill()
eu_weights_aligned.plot(ax=axes[5], kind='area', stacked=True, colormap='tab10', alpha=0.7)
axes[5].set_title('EU Internal Signal Allocation (Softmax Weights)')
axes[5].set_ylabel('Signal Weight')
axes[5].set_ylim(0, 1)
axes[5].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.tight_layout()
plt.show()
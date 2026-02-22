import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import os

from data_loader import PipelineDataLoader, EuropeanDataLoader
from signals import ShortTermSignalGenerator, LongTermSignalGenerator, PCASignalGenerator, RobustRegressionBlender, RegimePCAHMMGenerator
from portfolio import PortfolioConstructor, CurrencyNeutralPortfolioConstructor 
from backtester import Backtester
import config

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def run_regional_pipeline(region='US'):
    """Encapsulates the data, alpha, and portfolio generation for a region."""
    print(f"\n========================================")
    print(f"       STARTING {region} PIPELINE")
    print(f"========================================")
    
    if region == 'US':
        loader = PipelineDataLoader(benchmark_ticker='SPX')
        benchmark = 'SPX'
    else:
        loader = EuropeanDataLoader(benchmark_ticker='SX5E')
        benchmark = 'SX5E'
        
    data = loader.fetch_all()
    
    print(f"[{region}] Generating Alphas...")
    short_gen = ShortTermSignalGenerator(reversal_window=5)
    long_gen = LongTermSignalGenerator() 
    pca_gen = PCASignalGenerator(pca_update_freq=21)
    regime_gen = RegimePCAHMMGenerator(pca_update_freq=63, n_components=10)
    blender = RobustRegressionBlender(lookback=252, temperature=3)

    signal_dict = {
        "short": short_gen.generate(data.hedged_returns),
        "long": long_gen.generate(data.hedged_returns, data.earnings_yield, data.sectors),
        "pca": pca_gen.generate(data.hedged_returns),
        "regime": regime_gen.generate(data.hedged_returns)
    }

    priors = [0.25, 0.25, 0.25, 0.25]
    final_signals = blender.blend(signal_dict, data.hedged_returns, prior_weights=priors)
    
    print(f"[{region}] Constructing Portfolio...")
    adv_60d = data.volume_usd.rolling(window=60, min_periods=10).mean()
    
    # Currency dict is None for US, populated for EU
    curr_dict = getattr(data, 'currency_dict', None)
    if region == 'EU':
        portfolio_constructor = CurrencyNeutralPortfolioConstructor(
            target_ann_vol=config.PARAMS['TARGET_ANN_VOL'],
            max_adv_pct=config.PARAMS['MAX_ADV_PCT'],
            signal_threshold=0.75,     
            hard_volume_limit=2000000, 
            max_gross_exposure=10000000,
            currency_dict=curr_dict  
        )
    else:
        portfolio_constructor = PortfolioConstructor(
            target_ann_vol=config.PARAMS['TARGET_ANN_VOL'],
            max_adv_pct=config.PARAMS['MAX_ADV_PCT'],
            signal_threshold=0.75,     
            hard_volume_limit=2000000, 
            max_gross_exposure=10000000
        )

    all_target_positions = pd.DataFrame(0.0, index=data.price_ret.index, columns=data.price_ret.columns)
    current_positions = pd.Series(0.0, index=data.price_ret.columns)
    warmup = 252 + 60

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
                adv_60d=adv_60d.loc[t], betas=data.betas.loc[t], benchmark_ticker=benchmark
            )
        else:
            daily_total_ret = data.price_ret.loc[t].fillna(0) + data.div_ret.loc[t].fillna(0)
            current_positions = current_positions * (1 + daily_total_ret)

        all_target_positions.loc[t] = current_positions

    return data, all_target_positions, loader


# =================================================================================
# 1. RUN INDIVIDUAL PIPELINES
# =================================================================================
us_data, us_positions, us_loader = run_regional_pipeline('US')
eu_data, eu_positions, eu_loader = run_regional_pipeline('EU')

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
temperature = 0.05 
exp_us = np.exp(us_roll_sharpe / temperature)
exp_eu = np.exp(eu_roll_sharpe / temperature)

dynamic_weight_us = exp_us / (exp_us + exp_eu)
dynamic_weight_eu = exp_eu / (exp_us + exp_eu)

# --- NEW: BAYESIAN PRIOR BLENDING ---
# 1. Define your structural base weights
prior_us = 0.7
prior_eu = 0.3

# 2. Define how strongly you trust the prior vs. the dynamic momentum (0.0 to 1.0)
#    0.0 = Fully dynamic (ignores prior), 1.0 = Fully static (pegs to prior)
prior_confidence = 0.5

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

# =================================================================================
# 4. LIVE EXECUTION EXPORT (US AND EU)
# =================================================================================
print("\n========================================")
print("         PHASE 4: LIVE EXECUTION EXPORT")
print("========================================")
last_date = common_dates[-1]
print(f"Valid global signals generated for date: {last_date.date()}")
print(f"Current Global Allocation -> US: {weight_us.iloc[-1]:.1%}, EU: {weight_eu.iloc[-1]:.1%} (Scale Factor: {vol_scale_factor.iloc[-1]:.2f}x)")

# --- US EXPORT ---
us_active = us_global_final.iloc[-1]
us_active = us_active[us_active != 0].copy()
us_exec = pd.DataFrame({
    'internal_code': us_active.index,
    'currency': 'USD',
    'target_notional': us_active.values.round(2)
})
us_exec.to_csv('target_notionals_us_t_plus_1.csv', index=False)
print(f"Saved {len(us_exec)} US target positions.")

# --- EU EXPORT (With FX Translation) ---
eu_active = eu_global_final.iloc[-1]
eu_active = eu_active[eu_active != 0].copy()

fx_multipliers = eu_loader.processor.process_fx(config.EU_FX_FILE)
latest_fx = fx_multipliers.reindex(eu_data.price_ret.index).ffill().loc[last_date]

eu_exec = pd.DataFrame({
    'internal_code': eu_active.index,
    'currency': [eu_data.currency_dict.get(ric, 'EUR').upper() for ric in eu_active.index], 
    'target_notional_usd': eu_active.values
})

def get_local_notional_fixed(row):
    raw_curr = row['currency']
    fx_col = f"{raw_curr}="
    
    # Check if we have the specific rate, otherwise fallback to EUR
    rate = latest_fx.get(fx_col, latest_fx.get('EUR=', 1.0))
    local_val = row['target_notional_usd'] / rate
    
    # PER USER INSTRUCTIONS: GBp is already correct as GBP value, so no /100 needed.
    return local_val

eu_exec['target_notional'] = eu_exec.apply(get_local_notional_fixed, axis=1).round(2)
eu_exec['currency'] = eu_exec['currency'].apply(lambda x: 'GBP' if x == 'GBP' else x) # Ensure clean label
eu_exec = eu_exec[['internal_code', 'currency', 'target_notional']]
eu_exec.to_csv('target_notionals_eu_t_plus_1.csv', index=False)
print(f"Saved {len(eu_exec)} EU target positions.")

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
us_bench = us_data.benchmark_series.reindex(common_dates).fillna(0)
eu_bench = eu_data.benchmark_series.reindex(common_dates).fillna(0)
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
print(f"  Cumulative PnL:       ${cum_pnl.iloc[-1]:,.2f}")
print(f"  Annualized PnL:       ${annualized_pnl:,.2f}")
print(f"  Annualized Vol:       ${annualized_vol:,.2f} (Target: $500,000)")
print(f"  Sharpe Ratio:         {sharpe:.3f}")
print(f"  Max Drawdown:         ${max_drawdown:,.2f}")

print(f"\n[ RISK & CORRELATION ]")
print(f"  Correlation vs SPX:   {corr_spx:.3f}")
print(f"  Correlation vs SX5E:  {corr_sx5e:.3f}")
print(f"  Avg Gross Exposure:   ${avg_gross:,.2f}")
print(f"  Avg Net Exposure:     ${avg_net:,.2f}")

print(f"\n[ FRICTION & EXECUTION ]")
print(f"  Annualized Turnover:  ${annual_turnover:,.2f} ({turnover_pct:.1f}% of Gross)")
print(f"  Total T-Costs:        ${total_tcosts:,.2f}")
print(f"  Total Financing:      ${total_financing:,.2f}")
print(f"  Total Friction Drag:  ${(total_tcosts + total_financing):,.2f}")


# --- PLOTTING ---
fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

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

# Plot rolling sharpe:
rolling_sharpe = np.sqrt(252) * net_pnl.rolling(window=252).mean() / net_pnl.rolling(window=63).std()
rolling_sharpe.plot(ax=axes[3], color='orange', lw=2)
axes[3].set_title('Rolling 63-Day Sharpe Ratio')
axes[3].axhline(0, color='black', ls='--', alpha=0.4)
axes[3].axhline(1, color='green', ls='--', alpha=0.4)
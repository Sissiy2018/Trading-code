import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import os

# 1. Load the extensions (useful if running in Jupyter/IPython)
try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '3')
except NameError:
    pass # Ignore if running as a standard python script

# 2. Import our modules
import config
from data_loader import EuropeanDataLoader
from signals import ShortTermSignalGenerator, LongTermSignalGenerator, PCASignalGenerator, RobustRegressionBlender, RegimePCAHMMGenerator
# Note: Ensure CurrencyNeutralPortfolioConstructor is saved in your portfolio.py or portfolio_eu.py
from portfolio import CurrencyNeutralPortfolioConstructor 
from backtester import Backtester

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Data Pipeline ---
print("========================================")
print("       PHASE 1: EU DATA INGESTION (USD)")
print("========================================")
# The European loader handles FX mapping and USD translations automatically
loader = EuropeanDataLoader(benchmark_ticker='SX5E')
data = loader.fetch_all()

# --- 2. Alpha Generation ---
print("\n========================================")
print("       PHASE 2: ALPHA GENERATION")
print("========================================")
# Instantiate Generators (Running on the USD-converted returns!)
short_gen = ShortTermSignalGenerator(reversal_window=5)
long_gen = LongTermSignalGenerator() 
pca_gen = PCASignalGenerator(pca_update_freq=21)
regime_gen = RegimePCAHMMGenerator(pca_update_freq=63, n_components=10)
blender = RobustRegressionBlender(lookback=252, temperature=3)

short_signals = short_gen.generate(data.hedged_returns)
long_signals = long_gen.generate(data.hedged_returns, data.earnings_yield, data.sectors)
pca_signals = pca_gen.generate(data.hedged_returns)
regime_signals = regime_gen.generate(data.hedged_returns)

signal_dict = {
    "short": short_signals,
    "long": long_signals,
    "pca": pca_signals,
    "regime": regime_signals
}

priors = [0.2, 0.2, 0.3, 0.3]
# Blend them via 60-Day Robust Regression
final_signals = blender.blend(signal_dict, data.hedged_returns, prior_weights=priors)
print(f"  Signal coverage: {final_signals.notna().any(axis=1).sum()} / {len(final_signals)} days")

# --- 3. Iterative Portfolio Construction ---
print("\n========================================")
print("     PHASE 3: CURRENCY-NEUTRAL PORTFOLIO")
print("========================================")

adv_60d = data.volume_usd.rolling(window=60, min_periods=10).mean()

# Instantiate the EU-specific constructor 
portfolio_constructor = CurrencyNeutralPortfolioConstructor(
    target_ann_vol=config.PARAMS['TARGET_ANN_VOL'],
    max_adv_pct=config.PARAMS['MAX_ADV_PCT'],
    signal_threshold=0.75,     
    hard_volume_limit=2000000, 
    max_gross_exposure=10000000,
    currency_dict=data.currency_dict  # CRITICAL: Pass the currency map here
)

all_target_positions = pd.DataFrame(0.0, index=data.price_ret.index, columns=data.price_ret.columns)
current_positions = pd.Series(0.0, index=data.price_ret.columns)

exposure_log = []
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
            
        # Fast covariance computation on active subset
        cov_matrix_small = data.tot_ret_clean[active_assets].loc[:t].iloc[-60:].cov()
        
        cov_matrix = cov_matrix_small.reindex(
            index=data.price_ret.columns, 
            columns=data.price_ret.columns, 
            fill_value=0.0
        )
        
        adv_t = adv_60d.loc[t]
        beta_t = data.betas.loc[t]

        # Generate new target portfolio
        current_positions = portfolio_constructor.generate_target_positions(
            t=t, 
            signals=sig_t, 
            cov_matrix=cov_matrix, 
            adv_60d=adv_t, 
            betas=beta_t, 
            benchmark_ticker='SX5E'
        )
        
    else:
        # Drift with daily USD returns
        daily_total_ret = data.price_ret.loc[t].fillna(0) + data.div_ret.loc[t].fillna(0)
        current_positions = current_positions * (1 + daily_total_ret)

    all_target_positions.loc[t] = current_positions
    
    # Exposure tracking
    assets_only = current_positions.index.difference(['SX5E'])
    gross_asset_exposure = current_positions[assets_only].abs().sum()
    net_asset_dollar = current_positions[assets_only].sum()
    benchmark_pos = current_positions.get('SX5E', 0.0)
    
    current_beta = data.betas.loc[t, assets_only].fillna(1.0)
    realized_beta = (current_positions[assets_only] * current_beta).sum()
    
    exposure_log.append({
        'Date': t,
        'Gross Asset Exposure': gross_asset_exposure,
        'Net Asset Dollar': net_asset_dollar,
        'Benchmark Position': benchmark_pos,
        'Net Portfolio Beta': realized_beta + benchmark_pos 
    })


# --- 4. Live Execution Export ---
print("\n========================================")
print("         PHASE 4: LIVE EXECUTION (EU)")
print("========================================")

last_date = data.price_ret.index[-1]
print(f"Valid signals generated for date: {last_date.date()}")
print("Preparing target notionals for t+1 execution...")

active_targets = current_positions[current_positions != 0].copy()

execution_df = pd.DataFrame({
    'internal_code': active_targets.index,
    'currency': [data.currency_dict.get(ric, 'EUR') for ric in active_targets.index], # Add currency column
    'target_notional_usd': active_targets.values.round(2)
})

output_file = 'target_notionals_eu_t_plus_1.csv'
execution_df.to_csv(output_file, index=False)
print(f"Successfully saved {len(execution_df)} target positions to {output_file}")


# --- 5. Exposure Plotting ---
exposure_df = pd.DataFrame(exposure_log).set_index('Date')

fig2, ax2 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

exposure_df[['Gross Asset Exposure', 'Net Asset Dollar', 'Benchmark Position']].plot(ax=ax2[0], lw=2)
ax2[0].set_title('EU Portfolio Exposures (USD)')
ax2[0].set_ylabel('Position Size ($)')
ax2[0].axhline(0, color='black', lw=1)
ax2[0].grid(True, alpha=0.3)

exposure_df['Net Portfolio Beta'].plot(ax=ax2[1], color='purple', lw=2)
ax2[1].set_title('Net Beta Exposure (Against SX5E in USD)')
ax2[1].set_ylabel('Beta-Adjusted Dollars')
ax2[1].axhline(0, color='black', lw=1)
ax2[1].grid(True, alpha=0.3)

plt.savefig('exposure_check_eu.png', dpi=150, bbox_inches='tight')
print("\nExposure plot saved to exposure_check_eu.png")


# --- 6. Backtesting Engine ---
print("\n========================================")
print("       PHASE 5: BACKTEST & REPORTING")
print("========================================")
backtester = Backtester(
    benchmark_ticker='SX5E', 
    tcost_bps=config.PARAMS['TCOST_BPS'], 
    div_tax_rate=config.PARAMS['DIV_TAX']
)
results = backtester.run(data.price_ret, data.div_ret, all_target_positions)

net_pnl = results['Net PnL']
annualized_pnl = net_pnl.mean() * 252
annualized_vol = net_pnl.std() * np.sqrt(252)
sharpe = annualized_pnl / annualized_vol if annualized_vol > 0 else 0
total_pnl = results['Cumulative PnL'].iloc[-1]

print(f"\n  Cumulative PnL:  ${total_pnl:,.2f}")
print(f"  Annualized PnL:  ${annualized_pnl:,.2f}")
print(f"  Annualized Vol:  ${annualized_vol:,.2f}")
print(f"  Sharpe Ratio:    {sharpe:.3f}")
print(f"  Total T-Costs:   ${results['T-Costs'].sum():,.2f}")
print(f"  Total Financing: ${results['Financing'].sum():,.2f}")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

results['Cumulative PnL'].plot(ax=axes[0], color='steelblue', lw=2)
axes[0].set_title('Cumulative PnL (European Book in USD)')
axes[0].set_ylabel('USD')
axes[0].axhline(0, color='red', ls='--', alpha=0.4)
axes[0].grid(True, alpha=0.3)

roll_sharpe = (net_pnl.rolling(252).mean() * 252) / (net_pnl.rolling(252).std() * np.sqrt(252))
pd.Series(roll_sharpe.values, index=data.price_ret.index).plot(ax=axes[1], color='darkorange', lw=2)
axes[1].set_title('Rolling 1-Year Sharpe')
axes[1].set_ylabel('Sharpe')
axes[1].axhline(0, color='red', ls='--', alpha=0.4)
axes[1].axhline(1, color='green', ls='--', alpha=0.4)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_results_eu.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to backtest_results_eu.png")
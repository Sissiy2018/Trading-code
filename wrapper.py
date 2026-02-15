import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Import the classes we wrote above
from preprocessing import DataProcessor
from signals import Momentum12_1M
from portfolio import PortfolioConstructor
from backtester import Backtester

def main():
    # --- 1. Pipeline Parameters ---
    BENCHMARK = 'SPX'
    REBALANCE_FREQ_DAYS = 21 # e.g., ~1 month
    TARGET_ANN_VOL = 500000
    MAX_ADV_PCT = 0.025
    TCOST_BPS = 3
    DIV_TAX = 0.30

    # Define file paths (adjust to your local paths)
    path = '.' 
    file_paths = [
        os.path.join(path, 'hist_data', 'data_1.csv'),
        # add other files here...
    ]

    # --- 2. Data Preprocessing ---
    print("Processing Data...")
    processor = DataProcessor(benchmark_ticker=BENCHMARK)
    price_close, price_ret, tot_ret, div_ret, volume_usd = processor.load_and_pivot(file_paths)
    
    # Impute and clean
    price_close_imp, price_ret_imp = processor.impute_missing(price_close)
    tot_ret_imp = tot_ret.fillna(0) 
    
    price_ret_clean = processor.clean_outliers(price_ret_imp)
    tot_ret_clean = processor.clean_outliers(tot_ret_imp)
    
    # Calculate Betas and Hedged Returns
    betas, hedged_returns = processor.compute_beta_and_hedge(tot_ret_clean, price_ret_clean)

    # --- 3. Signal Generation ---
    print("Generating Signals...")
    signal_gen = Momentum12_1M()
    signals = signal_gen.get_signals(hedged_returns)

    # Calculate rolling 60d ADV
    adv_60d = volume_usd.rolling(window=60, min_periods=10).mean()

    # --- 4. Iterative Portfolio Construction ---
    print("Constructing Portfolio through time...")
    portfolio_constructor = PortfolioConstructor(target_ann_vol=TARGET_ANN_VOL, max_adv_pct=MAX_ADV_PCT)
    
    # Dataframe to store the target end-of-day positions
    all_target_positions = pd.DataFrame(0.0, index=price_ret.index, columns=price_ret.columns)
    current_positions = pd.Series(0.0, index=price_ret.columns)

    for i, t in enumerate(price_ret.index):
        # Need enough data for 12-1M momentum (252 days) and 60d covariance
        if i < 252: 
            continue
            
        if i % REBALANCE_FREQ_DAYS == 0:
            # Rebalance Day: Compute Covariance and formulate new positions
            cov_matrix = tot_ret_clean.loc[:t].iloc[-60:].cov()
            
            sig_t = signals.loc[t]
            adv_t = adv_60d.loc[t]
            beta_t = betas.loc[t]
            
            current_positions = portfolio_constructor.generate_target_positions(
                t, sig_t, cov_matrix, adv_t, beta_t, BENCHMARK
            )
        else:
            # Non-Rebalance Day: Positions drift with daily returns
            daily_total_ret = price_ret.loc[t].fillna(0) + div_ret.loc[t].fillna(0)
            current_positions = current_positions * (1 + daily_total_ret)
            
        all_target_positions.loc[t] = current_positions

    # --- 5. Backtesting Engine ---
    print("Running Backtest Engine...")
    backtester = Backtester(benchmark_ticker=BENCHMARK, tcost_bps=TCOST_BPS, div_tax_rate=DIV_TAX)
    results = backtester.run(price_ret, div_ret, all_target_positions)

    # --- 6. Output & Reporting ---
    print("\n--- Backtest Summary ---")
    annualized_pnl = results['Net PnL'].mean() * 252
    annualized_vol = results['Net PnL'].std() * np.sqrt(252)
    sharpe = annualized_pnl / annualized_vol if annualized_vol > 0 else 0
    
    print(f"Annualized PnL: ${annualized_pnl:,.2f}")
    print(f"Annualized Vol: ${annualized_vol:,.2f}")
    print(f"Sharpe Ratio:   {sharpe:.2f}")
    print(f"Total T-Costs:  ${results['T-Costs'].sum():,.2f}")
    
    # Plotting
    results[['Cumulative PnL']].plot(title='Strategy Net Cumulative PnL', figsize=(10,6))
    plt.ylabel('USD')
    plt.grid(True)
    plt.show()


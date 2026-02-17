import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt

from preprocessing import DataProcessor
from signals import ComplexMLSignal, SignalConfig
from portfolio import ComplexPortfolioConstructor
from backtester import Backtester

warnings.filterwarnings('ignore')


def load_sectors(file_path, tickers):
    """Load TRBC sector classification from static data CSV."""
    df = pd.read_csv(file_path)
    sectors = df.set_index('Instrument')['TRBC Economic Sector Name']
    sectors = sectors.reindex(tickers).fillna('UNKNOWN')
    return sectors


def main():
    # --- 1. Pipeline Parameters ---
    BENCHMARK = 'SPX'
    REBALANCE_FREQ_DAYS = 21
    TARGET_ANN_VOL = 500000
    MAX_ADV_PCT = 0.025
    TCOST_BPS = 2
    DIV_TAX = 0.30

    cfg = SignalConfig()

    # Define file paths
    path = os.path.join('.', 'Hist_data_Russel3000', 'History_price')
    file_paths = [
        os.path.join(path, 'lseg_historyprice_data_20170522_to_20151208_ADVfiltered.csv'),
        os.path.join(path, 'lseg_historyprice_data_20181102_to_20170522_ADVfiltered.csv'),
        os.path.join(path, 'lseg_historyprice_data_20200420_to_20181102_ADVfiltered.csv'),
        os.path.join(path, 'lseg_historyprice_data_20210930_to_20200420_ADVfiltered.csv'),
        os.path.join(path, 'lseg_historyprice_data_20230319_to_20210930_ADVfiltered.csv'),
        os.path.join(path, 'lseg_historyprice_data_20240828_to_20230320_ADVfiltered.csv'),
        os.path.join(path, 'lseg_historyprice_data_20260214_to_20240829.csv'),
    ]

    # --- 2. Data Preprocessing ---
    print("Processing Data...")
    processor = DataProcessor(benchmark_ticker=BENCHMARK)
    price_close, price_ret, tot_ret, div_ret, volume, volume_usd = processor.load_and_pivot(file_paths)

    # Load S&P 500 benchmark as a standalone Series, then inject into panels
    sp_path = os.path.join('.', 'Hist_data_Russel3000', 'S&P',
                           'lseg_historyprice_S&P500_20260215_to_20151209.csv')
    sp_df = pd.read_csv(sp_path)
    sp_df['Date'] = pd.to_datetime(sp_df['Date'])
    sp_df = sp_df.set_index('Date').sort_index()
    sp_prices = sp_df['0'].astype(float)

    # Keep a standalone benchmark Series for the signal generator
    benchmark_series = sp_prices.reindex(price_close.index).copy()
    benchmark_series.name = BENCHMARK

    # Inject benchmark into the price/return panels
    price_close[BENCHMARK] = benchmark_series
    price_ret[BENCHMARK] = price_close[BENCHMARK].pct_change()
    tot_ret[BENCHMARK] = price_ret[BENCHMARK]
    div_ret[BENCHMARK] = 0.0
    volume[BENCHMARK] = 0.0
    volume_usd[BENCHMARK] = 0.0

    # Impute and clean
    price_close_imp, price_ret_imp, tot_ret_imp = processor.impute_missing(price_close, tot_ret)

    price_ret_clean = processor.clean_outliers(price_ret_imp)
    tot_ret_clean = processor.clean_outliers(tot_ret_imp)

    # Calculate Betas and Hedged Returns
    betas, hedged_returns = processor.compute_beta_and_hedge(tot_ret_clean, price_ret_clean)

    # --- Load Sector Data ---
    static_path = os.path.join('.', 'Hist_data_Russel3000', 'Static_data',
                               'lseg_static_data_20260216.csv')
    print("Loading TRBC sector data...")
    sectors = load_sectors(static_path, price_close_imp.columns)
    print(f"  Sectors: {sectors.nunique()} unique, {(sectors == 'UNKNOWN').sum()} unmapped")

    # --- 3. Signal Generation (Complex ML Pipeline) ---
    print("\nGenerating Signals (walk-forward ML ensemble)...")
    print("This may take several minutes...")
    signal_gen = ComplexMLSignal(cfg=cfg, random_state=42)
    signals = signal_gen.get_signals(
        hedged_returns,
        prices=price_close_imp,
        volume=volume_usd,
        benchmark=benchmark_series,
        sectors=sectors,
    )
    print(f"  Signal coverage: {signals.notna().any(axis=1).sum()} / {len(signals)} days")

    if signal_gen.diagnostics_.get('ensemble_weights_by_refit'):
        print(f"  Refit events: {len(signal_gen.diagnostics_['ensemble_weights_by_refit'])}")
        last_weights = list(signal_gen.diagnostics_['ensemble_weights_by_refit'].items())[-1]
        print(f"  Last ensemble: " + ", ".join(f"{k}={v:.2f}" for k, v in last_weights[1].items()))

    # Calculate rolling 60d ADV
    adv_60d = volume_usd.rolling(window=60, min_periods=10).mean()

    # --- 4. Iterative Portfolio Construction ---
    print("\nConstructing Portfolio through time...")
    portfolio_constructor = ComplexPortfolioConstructor(
        target_ann_vol=TARGET_ANN_VOL,
        max_adv_pct=MAX_ADV_PCT,
        max_pos_usd=cfg.max_pos_usd,
        name_cap_weight=cfg.name_cap_weight,
    )

    all_target_positions = pd.DataFrame(0.0, index=price_ret.index, columns=price_ret.columns)
    current_positions = pd.Series(0.0, index=price_ret.columns)

    # Warm-up: need train_window + lookback days before signals are valid
    warmup = max(cfg.train_window, cfg.beta_window, cfg.high52_window) + 60

    for i, t in enumerate(price_ret.index):
        if i < warmup:
            continue

        if i % REBALANCE_FREQ_DAYS == 0:
            sig_t = signals.loc[t]

            # Skip if no valid signal on this date
            if sig_t.isna().all():
                all_target_positions.loc[t] = current_positions
                continue

            cov_matrix = tot_ret_clean.loc[:t].iloc[-60:].cov()
            adv_t = adv_60d.loc[t]
            beta_t = betas.loc[t]

            new_positions = portfolio_constructor.generate_target_positions(
                t, sig_t, cov_matrix, adv_t, beta_t, BENCHMARK
            )

            # Enforce daily trade cap: max traded per day = 2.5% of 60d ADV
            trade = new_positions - current_positions
            max_trade = adv_t.reindex(trade.index).fillna(0.0) * MAX_ADV_PCT
            # Benchmark hedge trades are unconstrained
            if BENCHMARK in max_trade.index:
                max_trade[BENCHMARK] = np.inf
            trade_clipped = trade.clip(lower=-max_trade, upper=max_trade)
            current_positions = current_positions + trade_clipped
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
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (Complex ML Pipeline)")
    print("=" * 60)

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
    print(f"  Gross Price PnL: ${results['Gross Price PnL'].sum():,.2f}")
    print(f"  Dividend PnL:    ${results['Dividend PnL'].sum():,.2f}")

    # Annual breakdown
    results_dated = results.copy()
    results_dated.index = price_ret.index
    yearly = results_dated['Net PnL'].resample('YE').sum()
    print("\n  Annual PnL:")
    for dt, pnl in yearly.items():
        print(f"    {dt.year}: ${pnl:,.2f}")

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    results_dated['Cumulative PnL'].plot(ax=axes[0], color='steelblue', lw=2)
    axes[0].set_title('Cumulative PnL (Complex ML Pipeline)')
    axes[0].set_ylabel('USD')
    axes[0].axhline(0, color='red', ls='--', alpha=0.4)
    axes[0].grid(True, alpha=0.3)

    results_dated['Net PnL'].plot(ax=axes[1], color='darkgreen', alpha=0.6)
    axes[1].set_title('Daily PnL')
    axes[1].set_ylabel('USD')
    axes[1].axhline(0, color='red', ls='--', alpha=0.4)
    axes[1].grid(True, alpha=0.3)

    roll_sharpe = (net_pnl.rolling(252).mean() * 252) / (net_pnl.rolling(252).std() * np.sqrt(252))
    pd.Series(roll_sharpe.values, index=price_ret.index).plot(ax=axes[2], color='darkorange', lw=2)
    axes[2].set_title('Rolling 1-Year Sharpe')
    axes[2].set_ylabel('Sharpe')
    axes[2].axhline(0, color='red', ls='--', alpha=0.4)
    axes[2].axhline(1, color='green', ls='--', alpha=0.4)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('backtest_results_complex_ml.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to backtest_results_complex_ml.png")
    plt.show()


if __name__ == '__main__':
    main()

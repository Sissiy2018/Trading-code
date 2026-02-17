import pandas as pd
import numpy as np


class Backtester:
    """Backtest engine matching QRT Academy simulation parameters.

    Key QRT rules:
      - Positions are constant dollar notional (not constant shares).
        Rebalancing to maintain notional is free of charge.
      - Execution cost: 2bps on intentional target changes only.
      - Financing cost: 0.5% annualised on GMV (gross market value).
      - Dividends: 70% received on longs, 100% paid on shorts.
      - Benchmark hedge trades are zero cost.
    """

    def __init__(self, benchmark_ticker, tcost_bps=2, div_tax_rate=0.30,
                 financing_rate=0.005):
        self.benchmark_ticker = benchmark_ticker
        self.tcost = tcost_bps / 10000
        self.div_tax_rate = div_tax_rate
        self.financing_rate = financing_rate  # 0.5% annualised

    def run(self, price_ret, div_ret, target_positions):
        # Previous day's target = today's opening position (constant notional)
        pos_t_minus_1 = target_positions.shift(1).fillna(0)

        # 1. Price PnL: position * return
        price_pnl = pos_t_minus_1 * price_ret

        # 2. Dividend PnL: 70% on longs, 100% on shorts
        div_pnl = pd.DataFrame(0.0, index=price_ret.index, columns=price_ret.columns)
        long_mask = pos_t_minus_1 > 0
        short_mask = pos_t_minus_1 < 0
        div_pnl[long_mask] = pos_t_minus_1[long_mask] * div_ret[long_mask] * (1 - self.div_tax_rate)
        div_pnl[short_mask] = pos_t_minus_1[short_mask] * div_ret[short_mask]

        # 3. Transaction costs: only on intentional target changes
        # QRT maintains constant dollar notional for free — only charge
        # when the *target* itself changes from one day to the next.
        target_change = target_positions.diff().fillna(target_positions.iloc[0:1])

        # Benchmark hedge trades are zero cost
        if self.benchmark_ticker in target_change.columns:
            target_change[self.benchmark_ticker] = 0.0

        tcosts_usd = target_change.abs() * self.tcost

        # 4. Financing cost: 0.5% annualised on GMV (daily = rate / 252)
        gmv = pos_t_minus_1.abs().sum(axis=1)
        daily_financing = gmv * (self.financing_rate / 252)

        # 5. Total PnL Aggregation
        daily_pnl = (price_pnl.sum(axis=1)
                     + div_pnl.sum(axis=1)
                     - tcosts_usd.sum(axis=1)
                     - daily_financing)

        results = pd.DataFrame({
            'Gross Price PnL': price_pnl.sum(axis=1),
            'Dividend PnL': div_pnl.sum(axis=1),
            'T-Costs': tcosts_usd.sum(axis=1),
            'Financing': daily_financing,
            'Net PnL': daily_pnl,
        })
        results['Cumulative PnL'] = results['Net PnL'].cumsum()

        return results

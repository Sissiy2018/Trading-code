import pandas as pd

class Backtester:
    def __init__(self, benchmark_ticker, tcost_bps=3, div_tax_rate=0.30):
        self.benchmark_ticker = benchmark_ticker
        self.tcost = tcost_bps / 10000
        self.div_tax_rate = div_tax_rate

    def run(self, price_ret, div_ret, target_positions):
        pos_t_minus_1 = target_positions.shift(1).fillna(0)
        
        # 1. Price PnL
        price_pnl = pos_t_minus_1 * price_ret
        
        # 2. Dividend PnL
        div_pnl = pd.DataFrame(0.0, index=price_ret.index, columns=price_ret.columns)
        long_mask = pos_t_minus_1 > 0
        short_mask = pos_t_minus_1 < 0
        div_pnl[long_mask] = pos_t_minus_1[long_mask] * div_ret[long_mask] * (1 - self.div_tax_rate)
        div_pnl[short_mask] = pos_t_minus_1[short_mask] * div_ret[short_mask]

        # 3. Transaction Costs
        drifted_positions = pos_t_minus_1 * (1 + price_ret + div_ret)
        trades = target_positions - drifted_positions
        
        # --- ZERO COST HEDGE ---
        if self.benchmark_ticker in trades.columns:
            trades[self.benchmark_ticker] = 0.0
            
        tcosts_usd = trades.abs() * self.tcost

        # 4. Total PnL Aggregation
        daily_pnl = price_pnl.sum(axis=1) + div_pnl.sum(axis=1) - tcosts_usd.sum(axis=1)
        
        results = pd.DataFrame({
            'Gross Price PnL': price_pnl.sum(axis=1),
            'Dividend PnL': div_pnl.sum(axis=1),
            'T-Costs': tcosts_usd.sum(axis=1),
            'Net PnL': daily_pnl
        })
        results['Cumulative PnL'] = results['Net PnL'].cumsum()
        
        return results
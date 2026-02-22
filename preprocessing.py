import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, benchmark_ticker):
        self.benchmark_ticker = benchmark_ticker

    def load_and_pivot(self, file_paths):
        """Loads a list of daily CSVs and pivots them into time series."""
        df_list = [pd.read_csv(f) for f in file_paths]
        raw_df = pd.concat(df_list, ignore_index=True)
        raw_df['Date'] = pd.to_datetime(raw_df['Date'])
        
        # Sort and drop duplicates from overlapping file date ranges
        raw_df = raw_df.sort_values(by=['Date', 'RIC'])
        raw_df = raw_df.drop_duplicates(subset=['Date', 'RIC'], keep='last')

        # Pivot tables
        price_close = raw_df.pivot(index='Date', columns='RIC', values='Price Close')
        tot_ret = raw_df.pivot(index='Date', columns='RIC', values='Daily Total Return') / 100
        volume = raw_df.pivot(index='Date', columns='RIC', values='Volume')
        
        # Derived series
        price_ret = price_close.pct_change()
        div_ret = tot_ret - price_ret
        volume_usd = volume * price_close

        return price_close, price_ret, tot_ret, div_ret, volume, volume_usd

    def impute_missing(self, price_close, tot_ret):
        """Forward fills prices. Missing returns become 0."""
        # Forward fill prices
        price_close_imputed = price_close.ffill()
        # Recalculate returns based on imputed prices
        price_ret_imputed = price_close_imputed.pct_change().fillna(0)
        tot_ret_imputed = tot_ret.ffill().fillna(0)
        return price_close_imputed, price_ret_imputed, tot_ret_imputed

    def clean_outliers(self, returns_df, window=60, threshold=3.5):
        """Shrinks returns > 3.5 standard deviations from 0."""
        # Calculate rolling standard deviation
        roll_std = returns_df.rolling(window=window, min_periods=10).std()
        
        # Create upper and lower bounds
        upper_bound = threshold * roll_std
        lower_bound = -threshold * roll_std
        
        # Clip the returns
        cleaned_returns = returns_df.clip(lower=lower_bound, upper=upper_bound)
        return cleaned_returns

    def compute_beta_and_hedge(self, tot_ret_clean, price_ret_clean):
        """Computes rolling beta and returns hedged series."""
        bench_price_ret = price_ret_clean[self.benchmark_ticker]
        
        # Rolling variance of benchmark
        bench_var_250 = bench_price_ret.rolling(window=250, min_periods=50).var()
        
        betas = pd.DataFrame(index=tot_ret_clean.index, columns=tot_ret_clean.columns)
        
        for col in tot_ret_clean.columns:
            if col == self.benchmark_ticker:
                betas[col] = 1.0
                continue
            # Rolling covariance
            cov = tot_ret_clean[col].rolling(window=250, min_periods=50).cov(bench_price_ret)
            
            # Beta formula: 0.2 + 0.8 * (Cov / Var)
            raw_beta = cov / bench_var_250
            betas[col] = 0.2 + 0.8 * raw_beta
            
        # Hedged returns = Asset Total Return - (Beta * Benchmark Price Return)
        # Note: Depending on strategy math, you might subtract Benchmark Total Return instead.
        hedged_returns = tot_ret_clean.sub(betas.mul(bench_price_ret, axis=0), fill_value=0)
        
        return betas, hedged_returns
    
    def load_and_pivot_pe(self, file_paths):
        """Loads daily PE CSVs, pivots them, and calculates Earnings Yield."""
        df_list = [pd.read_csv(f) for f in file_paths]
        raw_df = pd.concat(df_list, ignore_index=True)
        raw_df['Date'] = pd.to_datetime(raw_df['Date'])
        
        # Sort and drop duplicates
        raw_df = raw_df.sort_values(by=['Date', 'RIC'])
        raw_df = raw_df.drop_duplicates(subset=['Date', 'RIC'], keep='last')

        # Pivot to time series
        pe_df = raw_df.pivot(index='Date', columns='RIC', values='Price to Earning')
        
        # Forward fill missing days (fundamentals don't update every day)
        pe_df = pe_df.ffill()
        
        # Convert to Earnings Yield (1 / PE) to stabilize outliers
        # Replace 0s with NaN temporarily to avoid division by zero
        ey_df = 1 / pe_df.replace(0, np.nan)
        
        return pe_df, ey_df


class EuropeanDataProcessor(DataProcessor): # <-- Add inheritance here
    def __init__(self, benchmark_ticker='SX5E'):
        super().__init__(benchmark_ticker) # <-- Initialize the parent class
        
        # Define quote conventions: True if quoted as USD per 1 Local (Multiply)
        # False if quoted as Local per 1 USD (Divide/Invert)
        self.fx_is_multiplier = {
            'EUR=': True,  
            'GBp=': True,  
            'SEK=': False, 
            'DKK=': False, 
            'NOK=': False,
            'CHF=': False,
            'PLN=': False,
            'USD=': True   
        }

    def process_fx(self, fx_file):
        """Loads and standardizes FX rates into a pure USD Multiplier matrix."""
        fx_df = pd.read_csv(fx_file, index_col='Date', parse_dates=True)
        fx_df = fx_df.ffill() # Forward fill missing days
        
        usd_multipliers = pd.DataFrame(index=fx_df.index)
        
        for col in fx_df.columns:
            if self.fx_is_multiplier.get(col, True):
                usd_multipliers[col] = fx_df[col]
            else:
                usd_multipliers[col] = 1.0 / fx_df[col]
                
        return usd_multipliers

    def load_and_pivot_eu(self, asset_files, fx_df):
        """Loads European assets and instantly translates them to USD."""
        # 1. Load Asset Data
        df_list = [pd.read_csv(f) for f in asset_files]
        raw_df = pd.concat(df_list, ignore_index=True)
        raw_df['Date'] = pd.to_datetime(raw_df['Date'])
        
        # 2. Extract Currency Mapping
        # Keep a mapping of RIC -> Currency for the Portfolio Constructor later
        currency_map = raw_df[['RIC', 'Currency']].dropna().drop_duplicates(subset=['RIC'], keep='last')
        currency_dict = dict(zip(currency_map['RIC'], currency_map['Currency']))
        
        raw_df = raw_df.sort_values(by=['Date', 'RIC']).drop_duplicates(subset=['Date', 'RIC'], keep='last')

        # 3. Pivot Local Data
        price_local = raw_df.pivot(index='Date', columns='RIC', values='Price Close')
        tot_ret_local = raw_df.pivot(index='Date', columns='RIC', values='Daily Total Return') / 100
        volume_local = raw_df.pivot(index='Date', columns='RIC', values='Volume')
        
        # 4. Apply USD Translation
        price_usd = pd.DataFrame(index=price_local.index, columns=price_local.columns)
        volume_usd = pd.DataFrame(index=price_local.index, columns=price_local.columns)
        fx_returns = pd.DataFrame(index=price_local.index, columns=price_local.columns)
        
        for ric in price_local.columns:
            curr = currency_dict.get(ric, 'EUR') # Default to EUR if missing
            
            # The British Pence Fix: Divide local price by 100 if GBp
            if curr == 'GBp':
                price_local[ric] = price_local[ric] / 100.0
                
            fx_col = f"{curr}=" if f"{curr}=" in fx_df.columns else 'EUR='
            fx_multiplier_series = fx_df[fx_col].reindex(price_local.index).ffill()
            
            # Calculate USD metrics
            price_usd[ric] = price_local[ric] * fx_multiplier_series
            volume_usd[ric] = volume_local[ric] * price_local[ric] * fx_multiplier_series
            fx_returns[ric] = fx_multiplier_series.pct_change()
            
        # 5. Compound Returns: (1 + R_local) * (1 + R_fx) - 1
        price_ret_usd = price_usd.pct_change()
        tot_ret_usd = (1 + tot_ret_local) * (1 + fx_returns) - 1
        div_ret_usd = tot_ret_usd - price_ret_usd
        
        return price_usd, price_ret_usd, tot_ret_usd, div_ret_usd, volume_usd, currency_dict
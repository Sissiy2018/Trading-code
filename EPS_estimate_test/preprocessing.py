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
    
    def load_and_pivot_eps(self, file_paths, 
                       eps_mean_col='Earnings Per Share - SmartEstimate®',
                       eps_std_col='Earnings Per Share - Standard Deviation',
                       eps_pct_col='Earnings Per Share - Predicted Surprise PCT'):
        """Load EPS estimate CSVs, pivot mean/std time series, and optionally compute dispersion.

        Dispersion is defined as: Price / EPS Std.
        """
        df_list = [pd.read_csv(f) for f in file_paths]
        raw_df = pd.concat(df_list, ignore_index=True)
        raw_df['Date'] = pd.to_datetime(raw_df['Date'])

        # Sort and drop duplicates from overlapping file date ranges
        raw_df = raw_df.sort_values(by=['Date', 'RIC'])
        raw_df = raw_df.drop_duplicates(subset=['Date', 'RIC'], keep='last')

        # Pivot EPS estimate mean and standard deviation to time series
        eps_mean_df = raw_df.pivot(index='Date', columns='RIC', values=eps_mean_col)
        eps_std_df = raw_df.pivot(index='Date', columns='RIC', values=eps_std_col)
        eps_pct_df = raw_df.pivot(index='Date', columns='RIC', values=eps_pct_col)

        # Forward fill because estimate fields are not updated every day
        eps_mean_df = eps_mean_df.ffill()
        eps_std_df = eps_std_df.ffill()
        eps_pct_df = eps_pct_df.ffill()

    # Avoid division-by-zero later when building dispersion
    #eps_std_df = eps_std_df.replace(0, np.nan)
    #eps_pct_df = eps_pct_df.replace(0, np.nan)

        return eps_mean_df, eps_std_df, eps_pct_df



class EuropeanDataProcessor(DataProcessor): 
    def __init__(self, benchmark_ticker='SX5E'):
        super().__init__(benchmark_ticker) 
        
        self.fx_is_multiplier = {
            'EUR=': True,  
            'GBP=': True,  # Standardized to GBP
            'SEK=': False, 
            'DKK=': False, 
            'NOK=': False,
            'CHF=': False,
            'PLN=': False,
            'USD=': True   
        }

    def process_fx(self, fx_files):
        """Loads and standardizes FX rates into a pure USD Multiplier matrix."""
        df_list = []
        for fx_file in fx_files:
            df = pd.read_csv(fx_file, index_col='Date', parse_dates=True)
            df_list.append(df)
            
        # 1. Stack vertically (axis=0) instead of side-by-side
        fx_df = pd.concat(df_list, axis=0)
        
        # 2. Sort chronologically
        fx_df = fx_df.sort_index()
        
        # 3. Squash any duplicate dates and forward-fill missing values
        # This handles both time-split files AND currency-split files perfectly
        fx_df = fx_df.groupby(fx_df.index).last().ffill()
        
        # 4. Force dates to midnight to guarantee alignment later
        fx_df.index = pd.to_datetime(fx_df.index).normalize()

        usd_multipliers = pd.DataFrame(index=fx_df.index)

        for col in fx_df.columns:
            if self.fx_is_multiplier.get(col, True):
                usd_multipliers[col] = fx_df[col]
            else:
                usd_multipliers[col] = 1.0 / fx_df[col]
                
        return usd_multipliers

    def load_and_pivot_eu(self, asset_files, fx_df):
        """Loads European assets and instantly translates them to USD."""
        df_list = [pd.read_csv(f) for f in asset_files]
        raw_df = pd.concat(df_list, ignore_index=True)
        
        # 3. FIX: Strip time components so it perfectly matches fx_df.index
        raw_df['Date'] = pd.to_datetime(raw_df['Date']).dt.normalize()
        
        currency_map = raw_df[['RIC', 'Currency']].dropna().drop_duplicates(subset=['RIC'], keep='last')
        currency_dict = dict(zip(currency_map['RIC'], currency_map['Currency']))
        
        raw_df = raw_df.sort_values(by=['Date', 'RIC']).drop_duplicates(subset=['Date', 'RIC'], keep='last')

        price_local = raw_df.pivot(index='Date', columns='RIC', values='Price Close')
        tot_ret_local = raw_df.pivot(index='Date', columns='RIC', values='Daily Total Return') / 100
        volume_local = raw_df.pivot(index='Date', columns='RIC', values='Volume')
        
        price_usd = pd.DataFrame(index=price_local.index, columns=price_local.columns)
        volume_usd = pd.DataFrame(index=price_local.index, columns=price_local.columns)
        fx_returns = pd.DataFrame(index=price_local.index, columns=price_local.columns)
        
        for ric in price_local.columns:
            curr = currency_dict.get(ric, 'EUR') 
            
            # 4. FIX: Correctly map British Pence to British Pounds for the FX lookup
            if curr == 'GBp':
                price_local[ric] = price_local[ric] / 100.0
                curr = 'GBP' 
                
            fx_col = f"{curr}=" 
            
            # Add a safety warning so you know if data is missing, rather than failing silently
            if fx_col not in fx_df.columns:
                print(f"  [Warning] Missing FX rate for {fx_col}. Defaulting {ric} to EUR.")
                fx_col = 'EUR='
                
            fx_multiplier_series = fx_df[fx_col].reindex(price_local.index).ffill()
            
            price_usd[ric] = price_local[ric] * fx_multiplier_series
            volume_usd[ric] = volume_local[ric] * price_local[ric] * fx_multiplier_series
            fx_returns[ric] = fx_multiplier_series.pct_change()
            
        price_ret_usd = price_usd.pct_change()
        tot_ret_usd = (1 + tot_ret_local) * (1 + fx_returns) - 1
        div_ret_usd = tot_ret_usd - price_ret_usd
        
        return price_usd, price_ret_usd, tot_ret_usd, div_ret_usd, volume_usd, currency_dict
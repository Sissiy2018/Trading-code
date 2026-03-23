import pandas as pd
import numpy as np
from dataclasses import dataclass
from preprocessing import DataProcessor, EuropeanDataProcessor
import config_daily as config

@dataclass
class MarketData:
    """A clean container for all loaded and preprocessed pipeline data."""
    price_ret: pd.DataFrame
    div_ret: pd.DataFrame
    tot_ret_clean: pd.DataFrame
    volume_usd: pd.DataFrame
    hedged_returns: pd.DataFrame
    earnings_yield: pd.DataFrame
    betas: pd.DataFrame
    sectors: pd.Series
    benchmark_series: pd.Series
    currency_dict: dict = None  # <--- ADD THIS LINE!
    eps_estimate_mean: pd.DataFrame = None
    eps_estimate_std: pd.DataFrame = None
    eps_estimate_pct: pd.DataFrame = None

class PipelineDataLoader:
    def __init__(self, benchmark_ticker='SPX'):
        self.benchmark = benchmark_ticker
        self.processor = DataProcessor(benchmark_ticker=self.benchmark)

    def _load_sectors(self, file_path, tickers):
        """Helper to load TRBC sector classification."""
        df = pd.read_csv(file_path)
        sectors = df.set_index('Instrument')['TRBC Economic Sector Name']
        return sectors.reindex(tickers).fillna('UNKNOWN')

    def fetch_all(self):
        print("1/4 Loading and Pivoting Price Data...")
        price_close, price_ret, tot_ret, div_ret, volume, volume_usd = self.processor.load_and_pivot(config.PRICE_FILES)

        print("2/4 Injecting Benchmark Data...")
        for file in config.SP_FILES:
            sp_df = pd.read_csv(file)
            sp_df['Date'] = pd.to_datetime(sp_df['Date'])
            sp_df = sp_df.set_index('Date').sort_index()

            benchmark_series = sp_df['0'].astype(float).reindex(price_close.index).copy()
        benchmark_series.name = self.benchmark

        # Inject into panels
        price_close[self.benchmark] = benchmark_series
        price_ret[self.benchmark] = price_close[self.benchmark].pct_change()
        tot_ret[self.benchmark] = price_ret[self.benchmark]
        div_ret[self.benchmark] = 0.0
        volume_usd[self.benchmark] = 0.0

        print("3/4 Imputing, Cleaning Outliers, and Calculating Betas...")
        price_close_imp, price_ret_imp, tot_ret_imp = self.processor.impute_missing(price_close, tot_ret)
        price_ret_clean = self.processor.clean_outliers(price_ret_imp)
        tot_ret_clean = self.processor.clean_outliers(tot_ret_imp)
        
        betas, hedged_returns = self.processor.compute_beta_and_hedge(tot_ret_clean, price_ret_clean)
        
        # Sanitize matrix extremes
        volume_usd = volume_usd.fillna(0.0)
        hedged_returns = hedged_returns.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        print("4/4 Loading Fundamentals and Sector Data...")
        _, ey_raw = self.processor.load_and_pivot_pe(config.PE_FILES)
        earnings_yield = ey_raw.reindex(index=price_ret.index, columns=price_ret.columns).ffill().fillna(0)
        
        sectors = self._load_sectors(config.STATIC_FILE, price_close_imp.columns)
        print(f"    Sectors: {sectors.nunique()} unique, {(sectors == 'UNKNOWN').sum()} unmapped")

        return MarketData(
            price_ret=price_ret,
            div_ret=div_ret,
            tot_ret_clean=tot_ret_clean,
            volume_usd=volume_usd,
            hedged_returns=hedged_returns,
            earnings_yield=earnings_yield,
            betas=betas,
            sectors=sectors,
            benchmark_series=benchmark_series
        )


import pandas as pd
import numpy as np
from dataclasses import dataclass
from preprocessing import DataProcessor, EuropeanDataProcessor
import config_daily as config

@dataclass
class MarketData:
    """A clean container for all loaded and preprocessed pipeline data."""
    price_ret: pd.DataFrame
    div_ret: pd.DataFrame
    tot_ret_clean: pd.DataFrame
    volume_usd: pd.DataFrame
    hedged_returns: pd.DataFrame
    earnings_yield: pd.DataFrame
    betas: pd.DataFrame
    sectors: pd.Series
    benchmark_series: pd.Series
    currency_dict: dict = None
    eps_estimate_mean: pd.DataFrame = None
    eps_estimate_std: pd.DataFrame = None
    eps_estimate_pct: pd.DataFrame = None

class PipelineDataLoader:
    def __init__(self, benchmark_ticker='SPX'):
        self.benchmark = benchmark_ticker
        self.processor = DataProcessor(benchmark_ticker=self.benchmark)

    def _load_sectors(self, file_path, tickers):
        df = pd.read_csv(file_path)
        sectors = df.set_index('Instrument')['TRBC Economic Sector Name']
        return sectors.reindex(tickers).fillna('UNKNOWN')

    def fetch_all(self):
        print("1/4 Loading and Pivoting Price Data...")
        price_close, price_ret, tot_ret, div_ret, volume, volume_usd = self.processor.load_and_pivot(config.PRICE_FILES)

        print("2/4 Injecting Benchmark Data...")
        sps = []
        for file in config.SP_FILES:
            sp_df = pd.read_csv(file)
            sp_df.columns = sp_df.columns.str.strip() # Strip whitespace from headers
            sp_df['Date'] = pd.to_datetime(sp_df['Date'])
            sp_df = sp_df.set_index('Date')
            
            # Robustly grab the first column if '0' isn't explicitly there
            target_col = '0' if '0' in sp_df.columns else sp_df.columns[0]
            sps.append(sp_df[target_col].astype(float))

        # Concatenate Series directly and drop duplicate DATES, not values
        benchmark_raw = pd.concat(sps)
        benchmark_raw = benchmark_raw[~benchmark_raw.index.duplicated(keep='last')].sort_index()
        
        benchmark_series = benchmark_raw.reindex(price_close.index).copy()
        benchmark_series.name = self.benchmark

        # Inject into panels
        price_close[self.benchmark] = benchmark_series
        price_ret[self.benchmark] = price_close[self.benchmark].pct_change()
        tot_ret[self.benchmark] = price_ret[self.benchmark]
        div_ret[self.benchmark] = 0.0
        volume_usd[self.benchmark] = 0.0

        print("3/4 Imputing, Cleaning Outliers, and Calculating Betas...")
        price_close_imp, price_ret_imp, tot_ret_imp = self.processor.impute_missing(price_close, tot_ret)
        price_ret_clean = self.processor.clean_outliers(price_ret_imp)
        tot_ret_clean = self.processor.clean_outliers(tot_ret_imp)
        
        betas, hedged_returns = self.processor.compute_beta_and_hedge(tot_ret_clean, price_ret_clean)
        
        volume_usd = volume_usd.fillna(0.0)
        hedged_returns = hedged_returns.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        print("4/4 Loading Fundamentals and Sector Data...")
        _, ey_raw = self.processor.load_and_pivot_pe(config.PE_FILES)
        earnings_yield = ey_raw.reindex(index=price_ret.index, columns=price_ret.columns).ffill().fillna(0)
        
        sectors = self._load_sectors(config.STATIC_FILE, price_close_imp.columns)
        print(f"    Sectors: {sectors.nunique()} unique, {(sectors == 'UNKNOWN').sum()} unmapped")

        print("New: 5/5 Loading EPS Estimates and Calculating Statistics...")

        eps_mean_df, eps_std_df, eps_pct_df = self.processor.load_and_pivot_eps(config.EPS_EST_FILES)
        eps_estimate_std= eps_std_df.reindex(index=price_ret.index, columns=price_ret.columns).ffill().fillna(0)
        eps_estimate_mean = eps_mean_df.reindex(index=price_ret.index, columns=price_ret.columns).ffill().fillna(0)
        eps_estimate_pct = eps_pct_df.reindex(index=price_ret.index, columns=price_ret.columns).ffill().fillna(0)

        return MarketData(
            price_ret=price_ret,
            div_ret=div_ret,
            tot_ret_clean=tot_ret_clean,
            volume_usd=volume_usd,
            hedged_returns=hedged_returns,
            earnings_yield=earnings_yield,
            betas=betas,
            sectors=sectors,
            benchmark_series=benchmark_series,
            eps_estimate_mean=eps_estimate_mean,
            eps_estimate_std=eps_estimate_std,
            eps_estimate_pct=eps_estimate_pct
        )


class EuropeanDataLoader:
    def __init__(self, benchmark_ticker='SX5E'):
        self.benchmark = benchmark_ticker
        self.processor = EuropeanDataProcessor(benchmark_ticker=self.benchmark)

    def _load_sectors(self, file_path, tickers):
        df = pd.read_csv(file_path)
        sectors = df.set_index('Instrument')['TRBC Economic Sector Name']
        return sectors.reindex(tickers).fillna('UNKNOWN')

    def fetch_all(self):
        print("1/5 Loading and Processing FX Data...")
        fx_multipliers = self.processor.process_fx(config.EU_FX_FILES)

        print("2/5 Loading EU Price Data and converting to USD...")
        price_usd, price_ret, tot_ret, div_ret, volume_usd, currency_dict = self.processor.load_and_pivot_eu(
            config.EU_PRICE_FILES, fx_multipliers
        )

        print("3/5 Injecting SX5E Benchmark (Converting EUR to USD)...")
        sps = []
        for sp_file in config.EU_BENCHMARK_FILES:
            sp_df = pd.read_csv(sp_file)
            sp_df.columns = sp_df.columns.str.strip() 
            sp_df['Date'] = pd.to_datetime(sp_df['Date'], errors='coerce') # Handle bad dates
            sp_df = sp_df.dropna(subset=['Date']).set_index('Date')
            
            target_col = 'Close Price' if 'Close Price' in sp_df.columns else sp_df.columns[0]
            
            # THE FIX: Force numeric conversion and drop the metadata strings
            clean_series = pd.to_numeric(sp_df[target_col], errors='coerce').dropna()
            sps.append(clean_series)

        # Concatenate Series directly and explicitly drop duplicate dates
        benchmark_raw = pd.concat(sps)
        benchmark_raw = benchmark_raw[~benchmark_raw.index.duplicated(keep='last')].sort_index()

        # Multiply by EUR= FX rate
        eur_fx = fx_multipliers['EUR='].reindex(benchmark_raw.index).ffill()
        benchmark_close_usd = benchmark_raw * eur_fx 

        benchmark_series = benchmark_close_usd.reindex(price_usd.index).copy()
        benchmark_series.name = self.benchmark

        # Inject into panels
        price_usd[self.benchmark] = benchmark_series
        price_ret[self.benchmark] = price_usd[self.benchmark].pct_change()
        tot_ret[self.benchmark] = price_ret[self.benchmark]
        div_ret[self.benchmark] = 0.0
        volume_usd[self.benchmark] = 0.0

        print("4/5 Imputing, Cleaning Outliers, and Calculating USD Betas...")
        price_close_imp, price_ret_imp, tot_ret_imp = self.processor.impute_missing(price_usd, tot_ret)
        price_ret_clean = self.processor.clean_outliers(price_ret_imp)
        tot_ret_clean = self.processor.clean_outliers(tot_ret_imp)
        
        betas, hedged_returns = self.processor.compute_beta_and_hedge(tot_ret_clean, price_ret_clean)
        
        volume_usd = volume_usd.fillna(0.0)
        hedged_returns = hedged_returns.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        print("5/5 Loading EU Fundamentals and Sector Data...")
        _, ey_raw = self.processor.load_and_pivot_pe(config.EU_PE_FILES)
        earnings_yield = ey_raw.reindex(index=price_ret.index, columns=price_ret.columns).ffill().fillna(0)
        
        sectors = self._load_sectors(config.EU_STATIC_FILE, price_close_imp.columns)
        print(f"    Sectors: {sectors.nunique()} unique, {(sectors == 'UNKNOWN').sum()} unmapped")

        print("New: 6/6 Loading EPS Estimates and Calculating Statistics...")

        eps_mean_df, eps_std_df, eps_pct_df = self.processor.load_and_pivot_eps(config.EU_EPS_EST_FILES)
        eps_estimate_std= eps_std_df.reindex(index=price_ret.index, columns=price_ret.columns).ffill().fillna(0)
        eps_estimate_mean = eps_mean_df.reindex(index=price_ret.index, columns=price_ret.columns).ffill().fillna(0)
        eps_estimate_pct = eps_pct_df.reindex(index=price_ret.index, columns=price_ret.columns).ffill().fillna(0)


        return MarketData(
            price_ret=price_ret,
            div_ret=div_ret,
            tot_ret_clean=tot_ret_clean,
            volume_usd=volume_usd,
            hedged_returns=hedged_returns,
            earnings_yield=earnings_yield,
            betas=betas,
            sectors=sectors,
            benchmark_series=benchmark_series,
            currency_dict=currency_dict,
            eps_estimate_mean=eps_estimate_mean,
            eps_estimate_std=eps_estimate_std,
            eps_estimate_pct=eps_estimate_pct
        )
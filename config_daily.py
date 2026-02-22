import os

# --- 1. Trading & Pipeline Parameters ---
PARAMS = {
    'BENCHMARK': 'SPX',
    'REBALANCE_FREQ_DAYS': 2,
    'TARGET_ANN_VOL': 475000,
    'MAX_ADV_PCT': 0.025,
    'TCOST_BPS': 2,
    'DIV_TAX': 0.3
}

# --- 2. Base Paths ---
BASE_DIR = os.path.join('.', 'Hist_data_Russel3000')
PRICE_DIR = os.path.join(BASE_DIR, 'History_price')
NEW_DATA_DIR = os.path.join(BASE_DIR, 'Daily_new_data')
PE_DIR = os.path.join(BASE_DIR, 'History_PE')
SP_DIR = os.path.join(BASE_DIR, 'S&P')
STATIC_DIR = os.path.join(BASE_DIR, 'Static_data')

# --- 3. File Lists ---
PRICE_FILES = [
    os.path.join(PRICE_DIR, 'lseg_historyprice_data_20170522_to_20151208_ADVfiltered.csv'),
    os.path.join(PRICE_DIR, 'lseg_historyprice_data_20181102_to_20170522_ADVfiltered.csv'),
    os.path.join(PRICE_DIR, 'lseg_historyprice_data_20200420_to_20181102_ADVfiltered.csv'),
    os.path.join(PRICE_DIR, 'lseg_historyprice_data_20210930_to_20200420_ADVfiltered.csv'),
    os.path.join(PRICE_DIR, 'lseg_historyprice_data_20230319_to_20210930_ADVfiltered.csv'),
    os.path.join(PRICE_DIR, 'lseg_historyprice_data_20240828_to_20230320_ADVfiltered.csv'),
    os.path.join(PRICE_DIR, 'lseg_historyprice_data_20260214_to_20240829.csv'),
    os.path.join(NEW_DATA_DIR, 'lseg_historyprice_data_now_to_20260212_ADVfiltered.csv')
]

PE_FILES = [
    os.path.join(PE_DIR, 'lseg_Price-Earning_data_20170522_to_20151208_ADVfiltered.csv'),
    os.path.join(PE_DIR, 'lseg_Price-Earning_data_20181102_to_20170522_ADVfiltered.csv'),
    os.path.join(PE_DIR, 'lseg_Price-Earning_data_20181102_to_20170522_ADVfiltered.csv'),
    os.path.join(PE_DIR, 'lseg_Price-Earning_data_20200420_to_20181102_ADVfiltered.csv'),
    os.path.join(PE_DIR, 'lseg_Price-Earning_data_20200420_to_20210930_ADVfiltered.csv'),
    os.path.join(PE_DIR, 'lseg_Price-Earning_data_20211001_to_20230320_ADVfiltered.csv'),
    os.path.join(PE_DIR, 'lseg_Price-Earning_data_20260215_to_20230321_ADVfiltered.csv'),
    os.path.join(NEW_DATA_DIR, 'lseg_Price-Earning_data_now_to_20260212_ADVfiltered.csv')
]

SP_FILES = [
    os.path.join(SP_DIR, 'lseg_historyprice_S&P500_20260215_to_20151209.csv'),
    os.path.join(NEW_DATA_DIR, 'lseg_historyprice_S&P500_now_to_20260212.csv')
]
STATIC_FILE = os.path.join(STATIC_DIR, 'lseg_static_data_20260216.csv')

# ==========================================
# --- 4. European Pipeline Parameters ---
# ==========================================

EU_BASE_DIR = os.path.join('.', 'Hist_data_Stoxx600') # Change to your actual EU folder
EU_PRICE_DIR = os.path.join(EU_BASE_DIR, 'History_price')
EU_PE_DIR = os.path.join(EU_BASE_DIR, 'History_PE')
EU_STATIC_DIR = os.path.join(EU_BASE_DIR, 'Static_data')
NEW_EUR_DATA_DIR = os.path.join(EU_BASE_DIR, 'Daily_new_data')

# You will need to map these to your exact file names
EU_PRICE_FILES = [
    os.path.join(EU_PRICE_DIR, 'stoxx_historyprice_data_20161231_to_20150101.csv'), 
    os.path.join(EU_PRICE_DIR, 'stoxx_historyprice_data_20181231_to_20170101.csv'),
    os.path.join(EU_PRICE_DIR, 'stoxx_historyprice_data_20201231_to_20190101.csv'),
    os.path.join(EU_PRICE_DIR, 'stoxx_historyprice_data_20211231_to_20210101.csv'),
    os.path.join(EU_PRICE_DIR, 'stoxx_historyprice_data_20221231_to_20220101.csv'),
    os.path.join(EU_PRICE_DIR, 'stoxx_historyprice_data_20231231_to_20230101.csv'),
    os.path.join(EU_PRICE_DIR, 'stoxx_historyprice_data_20260219_to_20240101.csv'),
    os.path.join(NEW_EUR_DATA_DIR, 'stoxx_historyprice_data_now_to_20260219.csv')
]

EU_PE_FILES = [
    os.path.join(EU_PE_DIR, 'stoxx_Price-Earning_data_20151231_to_20150101.csv'),
    os.path.join(EU_PE_DIR, 'stoxx_Price-Earning_data_20161231_to_20160101.csv'),
    os.path.join(EU_PE_DIR, 'stoxx_Price-Earning_data_20171231_to_20170101.csv'),
    os.path.join(EU_PE_DIR, 'stoxx_Price-Earning_data_20181231_to_20180101.csv'),
    os.path.join(EU_PE_DIR, 'stoxx_Price-Earning_data_20191231_to_20190101.csv'),
    os.path.join(EU_PE_DIR, 'stoxx_Price-Earning_data_20211231_to_20200101.csv'),
    os.path.join(EU_PE_DIR, 'stoxx_Price-Earning_data_20231231_to_20220101.csv'),
    os.path.join(EU_PE_DIR, 'stoxx_Price-Earning_data_20260219_to_20240101.csv'),
    os.path.join(NEW_EUR_DATA_DIR, 'stoxx_Price-Earning_data_now_to_20260219.csv')
]

EU_FX_FILE = [os.path.join(EU_BASE_DIR, 'History_currency', 'currency_data_20260219_to_20150101.csv'),
              os.path.join(NEW_EUR_DATA_DIR, 'currency_data_now_to_20260219.csv')]
EU_BENCHMARK_FILES = [
    os.path.join(EU_BASE_DIR, 'Eurostoxx50','stoxx_historyprice_Eurostoxx50_20260220_to_20150101.csv'),
    os.path.join(NEW_EUR_DATA_DIR, 'stoxx_historyprice_Eurostoxx50_now_to_20260219.csv')
]
EU_STATIC_FILE = os.path.join(EU_STATIC_DIR, 'stoxx_static_data_20260220.csv')
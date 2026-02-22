import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

# 1. Load the extensions
%load_ext autoreload
%autoreload 3

# 2. Import our newly separated modules
import config
from data_loader import PipelineDataLoader
from signals import ShortTermSignalGenerator, LongTermSignalGenerator, PCASignalGenerator, RobustRegressionBlender, RegimePCAHMMGenerator
from portfolio import PortfolioConstructor
from backtester import Backtester

warnings.filterwarnings('ignore')

# --- 1. Data Pipeline ---
print("========================================")
print("       PHASE 1: DATA INGESTION")
print("========================================")
loader = PipelineDataLoader(benchmark_ticker=config.PARAMS['BENCHMARK'])
data = loader.fetch_all()

# --- 2. Alpha Generation ---
print("\n========================================")
print("       PHASE 2: ALPHA GENERATION")
print("========================================")
# Instantiate Generators
short_gen = ShortTermSignalGenerator(reversal_window=5)
long_gen = LongTermSignalGenerator() 
# pca_gen = PCASignalGenerator(pca_update_freq=21)
blender = RobustRegressionBlender(lookback=252*2, temperature=3)

# Generate isolated alpha streams
short_signals = short_gen.generate(data.hedged_returns)
long_signals = long_gen.generate(data.hedged_returns, data.earnings_yield, data.sectors)
pca_signals = pca_gen.generate(data.hedged_returns)
blender.historical_weights.plot()
# Blend them via 60-Day Robust Regression
final_signals = blender.blend(short_signals, long_signals, pca_signals, data.hedged_returns)
final_signals.iloc[:,:2].plot(figsize=(14, 6), lw=1.5, title='Sample of Final Blended Signals')
print(f"  Signal coverage: {final_signals.notna().any(axis=1).sum()} / {len(final_signals)} days")


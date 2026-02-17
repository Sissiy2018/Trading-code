import pandas as pd
import numpy as np

class SignalGenerator:
    def get_signals(self, **kwargs):
        raise NotImplementedError("Must implement get_signals")

class Momentum12_1M(SignalGenerator):
    def get_signals(self, hedged_returns):
        log_returns = np.log1p(hedged_returns)
        
        # Add min_periods! E.g., require at least 200 valid days out of 252
        mom_12m = log_returns.rolling(window=252, min_periods=200).sum()
        mom_1m = log_returns.rolling(window=21, min_periods=15).sum()
        
        signal = mom_12m - mom_1m
        return signal
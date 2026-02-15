import pandas as pd
import numpy as np

class SignalGenerator:
    def get_signals(self, **kwargs):
        raise NotImplementedError("Must implement get_signals")

class Momentum12_1M(SignalGenerator):
    def get_signals(self, hedged_returns):
        """
        Calculates 12-1 month momentum.
        Assumes 252 trading days in a year, 21 in a month.
        We sum the log returns from t-252 to t-21.
        """
        # Convert simple returns to log returns for additive summation
        log_returns = np.log1p(hedged_returns)
        
        # Sum over 252 days, subtract the sum over the last 21 days
        mom_12m = log_returns.rolling(window=252).sum()
        mom_1m = log_returns.rolling(window=21).sum()
        
        signal = mom_12m - mom_1m
        return signal
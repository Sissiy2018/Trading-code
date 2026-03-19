US={
    'short_term':
    {
        'horizon': 10,
        'reversal_window': 10,
        'smoothing_span': 2,
        'IC':0.073,
    },
    'long_term':
    {
        'horizon': 21,
        'momentum_window': 126,
        'skip_recent': 10,
        'smoothing_span': 5,
        'value_tilt_strength': 0.15,
        'IC':0.0729
    },
    'pca':
    {
        'horizon': 5,
        'n_components': 10,
        'cov_window': 252,
        'mom_window': 126,
        'rev_window': 10,
        'span': 10,
        'IC':0.0356
    },
    'defensive':
    {
        'horizon': 21,
        'drift_window': 42,
        'vol_window': 21,
        'smoothing_span': 5,
        'IC':0.0662
    },
    'hmm':
    {
        'horizon': 21,
        'n_components': 5,
        'pca_update_freq': 42,
        'max_states': 3,
        'hmm_window': 252,
        'IC': 0.0534
    },
    'volume':
    {
        'horizon': 21,
        'volume_window': 10,
        'return_window': 3,
        'smoothing_span': 3,
        'IC': 0.0634
    },
    'drift_regime':
    {
        'horizon': 21,
        'drift_window': 63,
        'drift_threshold': 0.60,
        'rev_window': 5,
        'value_weight':0.7,
        'smoothing_span': 3,
        'IC': -0.0434
    }
}
EU={
    'short_term':
    {
        'horizon': 10,
        'reversal_window': 7,
        'smoothing_span': 8,
        'IC':-0.0349
    },
    'long_term':
    {
        'horizon': 5,
        'momentum_window': 378,
        'skip_recent': 21,
        'smoothing_span': 10,
        'value_tilt_strength': 0.5,
        'IC':0.0531
    },
    'pca':
    {
        'horizon': 5,
        'n_components': 3,
        'cov_window': 126,
        'mom_window': 63,
        'rev_window': 21,
        'span': 10,
        'IC':0.0254
    },
      'defensive':
    {
        'horizon': 5,
        'drift_window': 42,
        'vol_window': 21,
        'smoothing_span': 5,
        'IC':-0.0369
    },
    'hmm':
    {
        'horizon': 21,
        'n_components': 5,
        'pca_update_freq': 42,
        'max_states': 5,
        'hmm_window': 500,
        'IC': 0.0353
    },
    'volume':
    {
        'horizon': 21,
        'volume_window': 10,
        'return_window': 3,
        'smoothing_span': 3,
        'IC': 0.0585
    },
    'drift_regime':
    {
        'horizon': 5,
        'drift_window': 63,
        'drift_threshold': 0.60,
        'rev_window': 10,
        'value_weight': 0.7,
        'smoothing_span': 5,
        'IC': 0.0304
    }
}
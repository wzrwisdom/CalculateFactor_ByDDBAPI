import numpy as np
from factor_cal.factor.basic_factor import register_facFunc
from scipy.stats import spearmanr, pearsonr

@register_facFunc('ret')
def ret(x:np.ndarray, shift:int=1) -> np.ndarray:
    x_shift = np.roll(x, shift, axis=0)
    x_shift[0:shift] = np.nan
    return x / x_shift - 1


@register_facFunc('roll_corr')
def rolling_correlation(x:np.ndarray, y:np.ndarray, window:int=10) -> np.ndarray:
    n_obs, n_features = x.shape
    
    # Initialize the array for storing the rolling correlation coefficients
    rolling_corrs = np.full((n_obs, n_features), np.nan)
    
    for i in range(n_obs - window + 1):
        x_window = x[i:i+window, :]
        y_window = y[i:i+window, :]
        
        for j in range(n_features):
            x_col = x_window[:, j]
            y_col = y_window[:, j]
            
            if np.std(x_col) != 0 and np.std(y_col) != 0:
                corr = pearsonr(x_col, y_col)[0]
            else:
                corr = np.nan
                
            rolling_corrs[i+window-1, j] = corr
    
    return rolling_corrs

@register_facFunc('roll_spear_corr')
def rolling_spearman_correlation(x:np.ndarray, y:np.ndarray, window:int=10) -> np.ndarray:
    n_obs, n_features = x.shape
    
    # Initialize the array for storing the rolling correlation coefficients
    rolling_corrs = np.full((n_obs, n_features), np.nan)
    
    for i in range(n_obs - window + 1):
        x_window = x[i:i+window, :]
        y_window = y[i:i+window, :]
        
        for j in range(n_features):
            x_col = x_window[:, j]
            y_col = y_window[:, j]
            
            if np.std(x_col) != 0 and np.std(y_col) != 0:
                corr = spearmanr(x_col, y_col)[0]
            else:
                corr = np.nan
                
            rolling_corrs[i+window-1, j] = corr
    
    return rolling_corrs
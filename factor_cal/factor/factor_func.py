import numpy as np
from factor_cal.factor.basic_factor import register_facFunc
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from typing import Union

@register_facFunc('ret')
def ret(x:np.ndarray, shift:int=1) -> np.ndarray:
    x_shift = np.roll(x, shift, axis=0)
    x_shift[0:shift] = np.nan
    return x / x_shift - 1

@register_facFunc('delta')
def delta(x:np.ndarray, shift:int=1) -> np.ndarray:
    x_shift = np.roll(x, shift, axis=0)
    x_shift[0:shift] = np.nan
    return x - x_shift

@register_facFunc('inv')
def inv(x:np.ndarray) -> np.ndarray:
    return 1 / x

@register_facFunc('move')
def move(x:np.ndarray, shift:int=1) -> np.ndarray:
    x_shift = np.roll(x, shift, axis=0)
    x_shift[0:shift] = np.nan
    return x_shift
    
@register_facFunc('rowrank')
def rank(x:np.ndarray) -> np.ndarray:
    x = pd.DataFrame(x)
    return x.rank(axis=1, pct=True).to_numpy()

@register_facFunc('ts_rank')
def ts_rank(x:np.ndarray, window:int=10) -> np.ndarray:
    x_df = pd.DataFrame(x)
    res = x_df.rolling(window=window, min_periods=int(window/3)).rank(pct=True, numeric_only=True)
    return res.to_numpy()

@register_facFunc('ts_mean')
def ts_mean(x:np.ndarray, window:int=10) -> np.ndarray:
    x_df = pd.DataFrame(x)
    res = x_df.rolling(window=window, min_periods=int(window/3)).mean(numeric_only=True, engine='numba')
    return res.to_numpy()

@register_facFunc('ts_std')
def ts_std(x:np.ndarray, window:int=10) -> np.ndarray:
    x_df = pd.DataFrame(x)
    res = x_df.rolling(window=window, min_periods=int(window/3)).std(numeric_only=True, engine='numba')
    return res.to_numpy()

@register_facFunc('ts_skew')
def ts_skew(x:np.ndarray, window:int=10) -> np.ndarray:
    x_df = pd.DataFrame(x)
    res = x_df.rolling(window=window, min_periods=int(window/3)).skew(numeric_only=True)
    return res.to_numpy()

@register_facFunc('ts_kurt')
def ts_kurt(x:np.ndarray, window:int=10) -> np.ndarray:
    x_df = pd.DataFrame(x)
    res = x_df.rolling(window=window, min_periods=int(window/3)).kurt(numeric_only=True)
    return res.to_numpy()

@register_facFunc('ts_sum')
def ts_sum(x:np.ndarray, window:int=10) -> np.ndarray:
    x_df = pd.DataFrame(x)
    res = x_df.rolling(window=window, min_periods=int(window/3)).sum(numeric_only=True, engine='numba')
    return res.to_numpy()

@register_facFunc('ts_min')
def ts_min(x:np.ndarray, window:int=10) -> np.ndarray:
    x_df = pd.DataFrame(x)
    res = x_df.rolling(window=window, min_periods=int(window/3)).min(numeric_only=True)
    return res.to_numpy()

@register_facFunc('ts_max')
def ts_max(x:np.ndarray, window:int=10) -> np.ndarray:
    x_df = pd.DataFrame(x)
    res = x_df.rolling(window=window, min_periods=int(window/3)).max(numeric_only=True)
    return res.to_numpy()

@register_facFunc('ts_med')
def ts_med(x:np.ndarray, window:int=10) -> np.ndarray:
    x_df = pd.DataFrame(x)
    res = x_df.rolling(window=window, min_periods=int(window/3)).median(numeric_only=True)
    return res.to_numpy()

@register_facFunc('ts_count')
def ts_count(x:np.ndarray, window:int=10) -> np.ndarray:
    x_df = pd.DataFrame(x)
    res = x_df.rolling(window=window, min_periods=int(window/3)).count(numeric_only=True)
    return res.to_numpy()

@register_facFunc('log')
def log(x:np.ndarray) -> np.ndarray:
    return np.log(x)

@register_facFunc('div')
def divide(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return np.divide(x, y)

@register_facFunc('mul')
def multiply(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return np.multiply(x, y)

@register_facFunc('sub')
def subtract(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return np.subtract(x, y)

@register_facFunc('avoid_zero')
def avoid_zero(x:np.ndarray, small_num:float=1e-4) -> np.ndarray:
    return np.where(x == 0, small_num, x)

@register_facFunc('add')
def add(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return np.add(x, y)

@register_facFunc('ts_corr')
def ts_correlation(x:np.ndarray, y:np.ndarray, window:int=10) -> np.ndarray:
    x_df = pd.DataFrame(x)
    y_df = pd.DataFrame(y)

    rolling_corrs = x_df.rolling(window=window, min_periods=int(window/2)).corr(y_df)
    rolling_corrs.replace([np.inf, -np.inf], np.nan, inplace=True)
    rolling_corrs = rolling_corrs.to_numpy()
    return rolling_corrs

@register_facFunc('ffill_na')
def ffill_na(x:np.ndarray) -> np.ndarray:
    x_df = pd.DataFrame(x)
    return x_df.ffill().to_numpy()

@register_facFunc('fill_na')
def fill_na(x:np.ndarray, fill_num:float=0.) -> np.ndarray:
    return np.nan_to_num(x, copy=False, nan=fill_num)

@register_facFunc('fill_na_v2')
def fill_na_v2(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return np.where(np.isnan(x), y, x)

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

@register_facFunc('ohlc_rat')
def ohlc_ratio(o:np.ndarray, h:np.array, l:np.array, c:np.array, window:int=20) -> np.ndarray:
    o = pd.DataFrame(o)
    h = pd.DataFrame(h)
    l = pd.DataFrame(l)
    c = pd.DataFrame(c)
    
    o = o.rolling(window=window).apply(lambda x: x[0], raw=True)
    h = h.rolling(window=window).max()
    l = l.rolling(window=window).min()
    c = c.rolling(window=window).apply(lambda x: x[-1], raw=True)
    
    res = (c - o) / (h - l)
    res.where((h - l) != 0, 0, inplace=True)
    return res.to_numpy()

@register_facFunc('close_adjusted')
def close_adjusted(close:np.ndarray , window:int=20) -> np.ndarray:
    return divide(subtract(close, ts_mean(close, window)), ts_std(close, window) )


@register_facFunc('avg_price')
def avg_price(price:np.ndarray, vol:np.ndarray, window:int=10) -> np.ndarray:
    return divide(ts_sum(multiply(price, vol), window), ts_sum(vol, window))


@register_facFunc('best_v_imbalance')
def best_v_imbalance(bv:np.ndarray, b1:np.ndarray, sv:np.ndarray, s1:np.ndarray, shift:int=1) -> np.ndarray:
    bv = pd.DataFrame(bv)
    b1 = pd.DataFrame(b1)
    sv = pd.DataFrame(sv)
    s1 = pd.DataFrame(s1)
    
    
    b_flag1 = (b1 == b1.shift(shift))
    s_flag1 = (s1 == s1.shift(shift))
    b_flag2 = (b1 < b1.shift(shift))
    s_flag2 = (s1 > s1.shift(shift))
    
    bv_change = np.where(b_flag2, 0, np.where(b_flag1, bv-bv.shift(shift), bv))
    sv_change = np.where(s_flag2, 0, np.where(s_flag1, sv-sv.shift(shift), sv))
    res = bv_change - sv_change
    res[0:shift] = np.nan
    return res

@register_facFunc('bs_press')
def bs_press(press_buy: np.ndarray, press_sell: np.ndarray):
    log_buy = np.log(press_buy)
    log_buy[log_buy == -np.inf] = 0
    
    log_sell = np.log(press_sell)
    log_sell[log_sell == -np.inf] = 0
    return log_buy - log_sell

@register_facFunc('trade_info_in_price_region')
def trade_info_in_price_region(number, price, op:str='gt', window:int=5*20, perc:float=0.8):
    assert number.shape == price.shape, "The shape of number and price should be the same"
    p_df = pd.DataFrame(price)
    prank_df = p_df.rolling(window=window).rank(pct=True, numeric_only=True)
    n_df = pd.DataFrame(number)
    if op=='gt':
        n_in_pregion = n_df[prank_df > perc].rolling(window=window, min_periods=1).sum()
    elif op=='lt':
        n_in_pregion = n_df[prank_df < perc].rolling(window=window, min_periods=1).sum()
    else:
        raise ValueError("The 'op' parameter should be 'gt' or 'lt'")
    n_total = n_df.rolling(window=window, min_periods=1).sum()
    res = n_in_pregion / n_total
    return res.to_numpy()


@register_facFunc('trade_avgvol_in_price_region')
def trade_avgvol_in_price_price(vol, num, price, op:str='gt', window:int=5*20, perc:float=0.8):
    assert (num.shape == price.shape) and (vol.shape == price.shape), "The shape of vol, num and price should be the same"
    p_df = pd.DataFrame(price)
    prank_df = p_df.rolling(window=window).rank(pct=True, numeric_only=True)
    n_df = pd.DataFrame(num)
    vol_df = pd.DataFrame(vol)
    if op=='gt':
        n_in_pregion = n_df[prank_df > perc].rolling(window=window, min_periods=1).sum()
        v_in_pregion = vol_df[prank_df > perc].rolling(window=window, min_periods=1).sum()
    elif op=='lt':
        n_in_pregion = n_df[prank_df < perc].rolling(window=window, min_periods=1).sum()
        v_in_pregion = vol_df[prank_df < perc].rolling(window=window, min_periods=1).sum()
    else:
        raise ValueError("The 'op' parameter should be 'gt' or 'lt'")
    n_total = n_df.rolling(window=window, min_periods=1).sum()
    v_total = vol_df.rolling(window=window, min_periods=1).sum()
    
    numerator = np.where(v_in_pregion==0, 0, v_in_pregion / n_in_pregion)
    res = numerator / (v_total/ n_total)
    return res.to_numpy()

@register_facFunc("HCVOL")
def HCVOL(cur_price:np.ndarray, close:np.ndarray, volume:np.ndarray, window:int=20) -> np.ndarray:
    assert cur_price.shape == close.shape == volume.shape, "The shape of cur_price, close and volume should be the same"
    n_obs, n_stock = cur_price.shape
    # Initialize the array for storing the rolling information
    rolling_res = np.full((n_obs, n_stock), np.nan)
    
    for i in range(n_obs-window+1):
        p_window = cur_price[i:i+window, :]
        c_window = close[i:i+window, :]
        v_window = volume[i:i+window, :]
        new_v_window = np.where(p_window > c_window[-1], v_window, 0)
        rolling_res[i+window-1,:] = new_v_window.sum(axis=0) / v_window.sum(axis=0)
    return rolling_res

@register_facFunc("LCVOL")
def LCVOL(cur_price:np.ndarray, close:np.ndarray, volume:np.ndarray, window:int=20) -> np.ndarray:
    assert cur_price.shape == close.shape == volume.shape, "The shape of cur_price, close and volume should be the same"
    n_obs, n_stock = cur_price.shape
    # Initialize the array for storing the rolling information
    rolling_res = np.full((n_obs, n_stock), np.nan)
    
    for i in range(n_obs-window+1):
        p_window = cur_price[i:i+window, :]
        c_window = close[i:i+window, :]
        v_window = volume[i:i+window, :]
        new_v_window = np.where(p_window < c_window[-1], v_window, 0)
        rolling_res[i+window-1,:] = new_v_window.sum(axis=0) / v_window.sum(axis=0)
    return rolling_res

@register_facFunc("HCP")
def HCP(cur_price:np.ndarray, close:np.ndarray, window:int=20) -> np.ndarray:
    assert cur_price.shape == close.shape, "The shape of cur_price and close should be the same"
    n_obs, n_stock = cur_price.shape
    # Initialize the array for storing the rolling information
    rolling_res = np.full((n_obs, n_stock), np.nan)
    
    for i in range(n_obs-window+1):
        p_window = cur_price[i:i+window, :]
        c_window = close[i:i+window, :]
        new_p_window = np.where(p_window > c_window[-1], p_window, np.nan)
        rolling_res[i+window-1,:] = np.nanmean(new_p_window, axis=0) / c_window[-1]
    return rolling_res

@register_facFunc("LCP")
def LCP(cur_price:np.ndarray, close:np.ndarray, window:int=20) -> np.ndarray:
    assert cur_price.shape == close.shape, "The shape of cur_price and close should be the same"
    n_obs, n_stock = cur_price.shape
    # Initialize the array for storing the rolling information
    rolling_res = np.full((n_obs, n_stock), np.nan)
    
    for i in range(n_obs-window+1):
        p_window = cur_price[i:i+window, :]
        c_window = close[i:i+window, :]
        new_p_window = np.where(p_window < c_window[-1], p_window, np.nan)
        rolling_res[i+window-1,:] = np.nanmean(new_p_window, axis=0) / c_window[-1]
    return rolling_res

@register_facFunc("LI")
def LI(close:np.ndarray, window:int=20) -> np.ndarray:
    return divide(ts_std(close, window), ts_mean(close, window))


@register_facFunc("bs_power_rough")
def bs_power_rough(buy_v:np.array, buy_p:np.array, sell_v:np.array, sell_p:np.array, close:np.array, window:int=20) -> np.ndarray:
    assert buy_v.shape == buy_p.shape == sell_v.shape == sell_p.shape, "The shape of buy_v, buy_p, sell_v and sell_p should be the same"
    n_obs, n_stock = buy_v.shape
    
    buy_p = np.nan_to_num(buy_p, nan=0)
    sell_p = np.nan_to_num(sell_p, nan=0)
    # Initialize the array for storing the rolling information
    rolling_res = np.full((n_obs, n_stock), np.nan)
    
    for i in range(n_obs-window+1):
        bv_window = buy_v[i:i+window, :]
        bp_window = buy_p[i:i+window, :]
        sv_window = sell_v[i:i+window, :]
        sp_window = sell_p[i:i+window, :]
        c = close[i+window-1, :]
        
        
        buy_power = bv_window * (bp_window / c)
        sell_power = sv_window * ((2*c - sp_window) / c)
        buy_power = np.nansum(buy_power, axis=0)
        sell_power = np.nansum(sell_power, axis=0)
        rolling_res[i+window-1,:] = (buy_power - sell_power) / (buy_power + sell_power + 1e-4)
    return rolling_res

def numpy_ffill(arr):
    arr = arr.T 
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out.T
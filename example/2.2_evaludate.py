import warnings
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ConstantInputWarning
import os, sys
sys.path.insert(0, "../")
from alphalens.utils import get_forward_returns_columns

from factor_cal.config_loader import basic_config as cfg
from factor_cal.table.ddb_table import PriceTable, SecLevelFacTable
from factor_cal.utils import ddb_utils as du

# igmore the warning of ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)
# obtain the ddb session
s = du.DDBSessionSingleton().session

class MaxLossExceededError(Exception):
    pass

def quantize_factor(factor_data, config, no_raise=False):
    def quantile_calc(x, _quantiles, _bins, _equal_quantile, _zero_aware, _no_raise):
        try:
            if _quantiles is not None and _bins is None and not _equal_quantile and not _zero_aware:
                return pd.qcut(x, _quantiles, labels=False) + 1
            elif _quantiles is not None and _bins is None and not _equal_quantile and _zero_aware:
                pos_quantiles = pd.qcut(x[x>=0], _quantiles // 2,
                                        labels=False) + _quantiles // 2 + 1
                neg_quantiles = pd.qcut(x[x<0], _quantiles // 2,
                                        labels=False) + 1
                return pd.concat([pos_quantiles, neg_quantiles]).sort_index()
            elif _quantiles is not None and _bins is None and _equal_quantile and not _zero_aware:
                nrow = x.shape[0]
                
                quantiles_list = []
                edges = [int(i) for i in np.linspace(0, nrow, _quantiles+1)]
                for i in range(_quantiles):
                    start, end = edges[i], edges[i+1]
                    quantiles_list += ([i+1] * (end - start))
                return pd.Series(quantiles_list, x.sort_values().index, name=x.name).sort_index()        
            elif _bins is not None and _quantiles is None and not _zero_aware:
                return pd.cut(x, _bins, labels=False) + 1
            elif _bins is not None and _quantiles is None and _zero_aware:
                pos_bins = pd.cut(x[x >= 0], _bins // 2,
                                  labels=False) + _bins // 2 + 1
                neg_bins = pd.cut(x[x < 0], _bins // 2,
                                  labels=False) + 1
                return pd.concat([pos_bins, neg_bins]).sort_index()
                
        except Exception as e:
            if _no_raise:
                return pd.Series(index=x.index)
            raise e
    
    grouper = [factor_data.index.get_level_values('datetime')]
    factor_quantile = factor_data.groupby(grouper)['factor'] \
        .apply(quantile_calc, config['quantiles'], config['bins'], config['equal_quantile'], config['zero_aware'], no_raise)
    factor_quantile.name = 'factor_quantile'
    return factor_quantile.dropna()

def get_clean_data(factor_and_ret, config):
    initial_amount = float(len(factor_and_ret.index))
    factor_and_ret.index = factor_and_ret.index.rename(['datetime', 'asset'])
    
    factor_and_ret = factor_and_ret.dropna()
    fwdret_amount = float(len(factor_and_ret.index))
    
    no_raise = False if config['max_loss'] == 0 else True
    quantile_data = quantize_factor(factor_and_ret, config, no_raise)
    
    factor_and_ret['factor_quantile'] = quantile_data
    
    factor_and_ret = factor_and_ret.dropna()
    
    binning_amount = float(len(factor_and_ret.index))
    
    tot_loss = (initial_amount - binning_amount) / initial_amount
    fwdret_loss = (initial_amount - fwdret_amount) / initial_amount
    bin_loss = tot_loss - fwdret_loss
    
    print("Dropped %.1f%% entries from factor data: %.1f%% in forward "
          "returns computation and %.1f%% in binning phase "
          "(set max_loss=0 to see potentially suppressed Exceptions)." %
          (tot_loss * 100, fwdret_loss * 100,  bin_loss * 100))
    
    if tot_loss > config['max_loss']:
        message = ("max_loss (%.1f%%) exceeded %.1f%%, consider increasing it."
                   % (config['max_loss'] * 100, tot_loss * 100))
        raise MaxLossExceededError(message)
    else:
        print("max_loss is %.1f%%, not exceeded: OK!" % (config['max_loss'] * 100))

    return factor_and_ret
    

def get_trade_close_bydate(stat_date, config):
    price_info = config['price_info']
    pc_tb = PriceTable(price_info['price_dbPath'], price_info['price_tbName'], price_info['time_col'], price_info['sec_col'], price_info['price_cols'])
    price = pc_tb.load_price(stat_date, config['start_time'], config['end_time'], sec_list=None)
    return s.loadTable(tableName=price).toDF()

def get_trade_return(close):
    close.set_index(['tradetime', 'securityid'], inplace=True)
    close.sort_index(inplace=True)
    res = close.groupby(close.index.get_level_values('securityid')).apply(lambda x: x.pct_change(1))
    res.rename(columns={'close': 'ret'}, inplace=True)
    return res

def load_factor_and_return(stat_date, factor_name, config):
    base_dirpath = '/home/wangzirui/workspace/data'
    factors_filepath = f'{base_dirpath}/fac_ret_{stat_date}.pkl'
    if not os.path.exists(factors_filepath):
        print(f'[warning]: There is no file:{factors_filepath}')
    factors_df = pd.read_pickle(factors_filepath)
    factors_df.set_index(['tradetime', 'securityid'], inplace=True)
    factors_df = factors_df.sort_index(level=0)
    
    # get the dataframe with factor and return information
    sel_cols = factors_df.columns[:3]
    sel_cols = np.append(sel_cols, [factor_name])
    factor_and_ret = factors_df[sel_cols].copy()
    factor_and_ret.rename(columns={factor_name: "factor"}, inplace=True)
    factor_and_ret.replace(np.inf, np.nan, inplace=True)
    
    # get trade close and return for each tradetime and each symbol
    close = get_trade_close_bydate(stat_date, config)
    td_ret = get_trade_return(close)
    
    # tidy the dataframe and quantile it
    factor_and_ret = get_clean_data(factor_and_ret, config['evaluation'])
    
    # calculate IC infomation in total or by group
    # ic_data = factor_information_coefficient(factor_and_ret)
    # ic_summary = get_factor_ic_summary_info(ic_data)
    
    # ic_data_bygroup = factor_information_coefficient(factor_and_ret, by_group=True)
    # ic_summary_bygroup = get_factor_ic_summary_info(ic_data_bygroup, by_group=True)
    
    # calculate pnl curve
    port_summary = factor_portfolio_return(factor_and_ret, td_ret, holding_time=20, long_short=True)
    port_summary['factor_name'] = factor_name
    
    
def form_portforlio_weight(factor_data, group_i, ngroup, holding_time, reverse=False):
    factor_data['wt'] = 0.
    keep_cols = ['factor', 'wt']
    if reverse:
        factor_data = factor_data[factor_data['factor_quantile'] == ngroup-group_i][keep_cols].copy()
    else:
        factor_data = factor_data[factor_data['factor_quantile'] == group_i][keep_cols].copy()
    
    grouper = [factor_data.index.get_level_values('datetime')]
    def cal_wt(group):
        group['wt'] = 1./group.shape[0]/holding_time
        return group
    factor_data = factor_data.groupby(grouper).apply(cal_wt)
    factor_data.sort_index(inplace=True)
    factor_data.index = factor_data.index.rename(['tradetime', 'securityid'])
    return factor_data

def form_portforlio_hedge_weight(factor_data, ngroup, holding_time, reverse=False):
    factor_data = factor_data.copy()
    factor_data['wt'] = 0
    keep_cols = ['factor', 'wt']
    
    grouper = [factor_data.index.get_level_values('datetime')]
    def cal_wt(group, dir=1.):
        group['wt'] = dir/group.shape[0]/holding_time
        return group
    
    first_group = factor_data['factor_quantile'] == 1
    last_group = factor_data['factor_quantile'] == ngroup
    
    if reverse:
        factor_data.loc[first_group, 'wt'] = factor_data[first_group].groupby(grouper).apply(cal_wt, dir=1.)
        factor_data.loc[last_group, 'wt'] = factor_data[last_group].groupby(grouper).apply(cal_wt, dir=-1.)
    else:
        factor_data.loc[first_group, 'wt'] = factor_data[first_group].groupby(grouper).apply(cal_wt, dir=-1.)
        factor_data.loc[last_group, 'wt'] = factor_data[last_group].groupby(grouper).apply(cal_wt, dir=1.)
    factor_data = factor_data.drop(factor_data[factor_data['wt']==0].index)
    factor_data.sort_index(inplace=True)
    factor_data.index = factor_data.index.rename(['tradetime', 'securityid'])
    return factor_data
    
    
def calc_stock_pnl(port_weight, ret_df, holding_time):
    port_weight = port_weight.reset_index()
    ret_df = ret_df.reset_index()

    last_ts_table = ret_df.groupby('securityid').apply(lambda x: max(x['tradetime']))
    last_ts_table.name='tradetime'
    last_ts_table = last_ts_table.reset_index()
    last_ts = dict(zip(last_ts_table['securityid'], last_ts_table['tradetime']))
    ages = np.arange(holding_time)
    
    time_stamps = port_weight['tradetime'].unique()
    dict_ts_index = {pd.Timestamp(ts): i for i, ts in enumerate(time_stamps)}
    dict_index_ts = {i: pd.Timestamp(ts) for i, ts in enumerate(time_stamps)}
    
    # 按照持有时间，将资金等分为相应份额
    pos = pd.merge(port_weight, pd.DataFrame({'age': ages}), how='cross')
    pos.rename(columns={'tradetime': 'tranche'}, inplace=True)
    pos['tradetime'] = pos.apply(lambda x: dict_index_ts[dict_ts_index[x['tranche']] + x['age']] \
                                if dict_index_ts[dict_ts_index[x['tranche']] + x['age']] < last_ts[x['securityid']] \
                                else np.nan, axis=1)
    pos = pos[(pos['tradetime'].notna())]
    
    pos = pos.merge(ret_df, on=['tradetime', 'securityid'], how='left')
    pos['cumret'] = pos.groupby(['securityid', 'tranche'])['ret'].transform(lambda x: np.cumprod(1 + x))
    pos['expr'] = pos['cumret'] * pos['wt']
    pos['pnl'] = pos['expr'] * pos['ret'] / (1 + pos['ret'])
    pos.loc[pos['age']==0, 'pnl'] = 0
    
    # 计算每个份额的净值
    tranche_info = pos.groupby('tranche')['pnl'].sum().reset_index().rename(columns={'pnl': 'tranche_pnl'})
    tranche_info['net_value'] = (1+tranche_info['tranche_pnl']).shift(20)
    tranche_info['net_value'].fillna(1, inplace=True)
    pos = pos.merge(tranche_info, on=['tranche'], how='left')
    pos['correct_pnl'] = pos['pnl'] * pos['net_value']
        
    return pos

def calc_portfolio_pnl(stock_pnl, col_name):
    return stock_pnl.groupby('tradetime')['correct_pnl'].sum().reset_index().rename(columns={'correct_pnl': col_name})

def evaluate_pnl(port_pnl, col_name):
    port_table = port_pnl.copy()
    port_table['net_value'] = 1+port_table[col_name].cumsum()
    port_table['ret_1min'] = port_table['net_value'].pct_change(20)
    
    mean_ret = port_table['ret'].mean()
    std_ret = port_table['ret'].std()
    sharpe_ratio = mean_ret / std_ret
    
    pos_ret_count = (port_table['ret'] > 0).sum()
    total_count = len(port_table)
    win_rate = pos_ret_count / total_count
    
    total_ret = port_table['net_value'].iloc[-1] - 1
    
    port_table['draw_down'] = (port_table['net_value'].cummax() - port_table['net_value']) / port_table['net_value'].cummax()
    max_draw_down = port_table['draw_down'].max()
    return pd.DataFrame({
        'group': [col_name],
        'total_ret': [total_ret],
        'mean_ret': [mean_ret],
        'std_ret': [std_ret],
        'sharpe_ratio': [sharpe_ratio],
        'win_rate': [win_rate],
        'max_draw_down': [max_draw_down]
    })
    
    


def factor_portfolio_return(factor_data, td_ret, holding_time=20, long_short=True):
    groups = np.sort(factor_data['factor_quantile'].unique())
    metric_summary = None
    
    # 分组回测
    for group_i in groups:
        portforlio_weight = form_portforlio_weight(factor_data, group_i, max(groups), holding_time)
        stock_pnl = calc_stock_pnl(portforlio_weight, td_ret, holding_time)
        portforlio_pnl = calc_portfolio_pnl(stock_pnl, 'group_'+group_i)
        metric_info = evaluate_pnl(portforlio_pnl)
        metric_summary = pd.concat([metric_summary, metric_info])
    
    # 多空组回测
    if (long_short):
        portforlio_weight = form_portforlio_hedge_weight(factor_data, max(groups), holding_time)
        stock_pnl = calc_stock_pnl(portforlio_weight, td_ret, holding_time)
        portforlio_pnl = calc_portfolio_pnl(stock_pnl, 'group_'+group_i)
        metric_info = evaluate_pnl(portforlio_pnl)
        metric_summary = pd.concat([metric_summary, metric_info])
        
    return metric_summary

def get_factor_ic_summary_info(ic_data, by_group=False):
    if by_group:
        grouper = ic_data.index.get_level_values('factor_quantile')
        
        ic_mean = ic_data.groupby(grouper).apply(lambda x: x.mean())
        ic_std = ic_data.groupby(grouper).apply(lambda x: x.std())
        icir = ic_mean/ic_std
        ic_summary_table = pd.concat([ic_mean, ic_std, icir], keys=['IC Mean', 'IC Std.', 'ICIR'])
        names = ic_summary_table.index.names
        names[0] = 'Type'
        ic_summary_table.index.rename(names)
        return ic_summary_table
    else:  
        ic_summary_table = pd.DataFrame()
        ic_summary_table["IC Mean"] = ic_data.mean()
        ic_summary_table["IC Std."] = ic_data.std()
        ic_summary_table["ICIR"] = \
            ic_data.mean() / ic_data.std()
        t_stat, p_value = stats.ttest_1samp(ic_data, 0, nan_policy='omit')
        ic_summary_table["t-stat(IC)"] = t_stat
        ic_summary_table["p-value(IC)"] = p_value
        ic_summary_table["IC Skew"] = stats.skew(ic_data, nan_policy='omit')
        ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data, nan_policy='omit')
        ic_summary_table['IC win rate'] = (ic_data > 0).sum() / ic_data.count()
        return ic_summary_table

def factor_information_coefficient(factor_data, by_group=False):
    def src_ic(group):
        f = group['factor']
        _ic = group[get_forward_returns_columns(factor_data.columns)] \
            .apply(lambda x: stats.spearmanr(x, f)[0])
        return _ic
    
    grouper = [factor_data.index.get_level_values('datetime')]
    
    if by_group:
        grouper.append('factor_quantile')
    
    ic = factor_data.groupby(grouper).apply(src_ic)
    return ic


if __name__ == "__main__":
    # read config file
    config = cfg.BasicConfig('config/config.yml')

    cur_date = '2023.09.22'
    factor_name = 'ret_v_prod_5min'
    load_factor_and_return(cur_date, factor_name, config)
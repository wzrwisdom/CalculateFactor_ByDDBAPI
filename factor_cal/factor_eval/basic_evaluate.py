from alphalens import performance as perf
from alphalens.utils import get_forward_returns_columns
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# def get_factor_ic_summary_info(data):
#     group_neutral = False
#     ic_data = perf.factor_information_coefficient(data, group_neutral)

#     ic_summary_table = pd.DataFrame()
#     ic_summary_table["IC Mean"] = ic_data.mean()
#     ic_summary_table["IC Std."] = ic_data.std()
#     ic_summary_table["ICIR"] = \
#         ic_data.mean() / ic_data.std()
#     t_stat, p_value = stats.ttest_1samp(ic_data, 0, nan_policy='omit')
#     ic_summary_table["t-stat(IC)"] = t_stat
#     ic_summary_table["p-value(IC)"] = p_value
#     ic_summary_table["IC Skew"] = stats.skew(ic_data, nan_policy='omit')
#     ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data, nan_policy='omit')
#     ic_summary_table['IC win rate'] = (ic_data > 0).sum() / ic_data.count()
#     return ic_summary_table


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

                x.name = 'value'
                tmp_x = x.reset_index()
                tmp_x_sorted = tmp_x.sort_values(by=['value', 'securityid'])
                x_sorted_index = pd.MultiIndex.from_frame(tmp_x_sorted[['tradetime', 'securityid']])
                
                return pd.Series(quantiles_list, x_sorted_index).sort_index()        
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
    grouper = [factor_data.index.get_level_values('tradetime')]
    factor_quantile = factor_data.groupby(grouper)['factor'] \
        .apply(quantile_calc, config['quantiles'], config['bins'], config['equal_quantile'], config['zero_aware'], no_raise)
    factor_quantile.name = 'factor_quantile'
    return factor_quantile.dropna()


def get_clean_data(factor_and_ret, config):
    factor_and_ret = factor_and_ret.copy()
    initial_amount = float(len(factor_and_ret.index))
    factor_and_ret.index = factor_and_ret.index.rename(['tradetime', 'securityid'])
    
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


def form_portforlio_weight(factor_data, group_i, ngroup, holding_time, reverse=False):
    if reverse:
        return factor_data[factor_data['factor_quantile'] == ngroup-group_i]
    else:
        return factor_data[factor_data['factor_quantile'] == group_i]
        
    # grouper = [factor_data.index.get_level_values('tradetime')]
    # def cal_wt(group):
    #     group['wt'] = 1./group.shape[0]/holding_time
    #     return group
    # factor_data = factor_data.groupby(grouper).apply(cal_wt)
    # factor_data.sort_index(inplace=True)
    # factor_data.index = factor_data.index.rename(['tradetime', 'securityid'])
    # return factor_data

def form_portforlio_hedge_weight(factor_data, ngroup, holding_time, reverse=False):
    if reverse:
        factor_data.loc[(factor_data['factor_quantile'] == ngroup), 'wt'] = -1\
            * factor_data.loc[(factor_data['factor_quantile'] == ngroup), 'wt'].abs()
        factor_data.loc[(factor_data['factor_quantile'] == 1), 'wt'] = 1\
            * factor_data.loc[(factor_data['factor_quantile'] == ngroup), 'wt'].abs()
    else:
        factor_data.loc[(factor_data['factor_quantile'] == 1), 'wt'] = -1\
            * factor_data.loc[(factor_data['factor_quantile'] == 1), 'wt'].abs()
        factor_data.loc[(factor_data['factor_quantile'] == ngroup), 'wt'] = 1\
            * factor_data.loc[(factor_data['factor_quantile'] == ngroup), 'wt'].abs()
    return factor_data[(factor_data['factor_quantile'] == 1) | (factor_data['factor_quantile'] == ngroup)]
    
    
def calc_stock_pnl(port_weight, ret_df, holding_time):
    port_weight = port_weight.reset_index()
    ret_df = ret_df.reset_index()

    # last_ts_table = ret_df.groupby('securityid').apply(lambda x: max(x['tradetime']))
    # last_ts_table.name='tradetime'
    # last_ts_table = last_ts_table.reset_index()
    # last_ts = dict(zip(last_ts_table['securityid'], last_ts_table['tradetime']))
    last_ts = min(ret_df['tradetime'].max(), port_weight['tradetime'].max())
    ages = [pd.Timedelta(i*3, unit='s') for i in np.arange(holding_time+1)]
    
    # 按照持有时间，将资金等分为相应份额
    pos = pd.merge(port_weight, pd.DataFrame({'age': ages}), how='cross')
    pos.rename(columns={'tradetime': 'tranche'}, inplace=True)
    
    # Method One
    # time_stamps = port_weight['tradetime'].unique()
    # dict_ts_index = {pd.Timestamp(ts): i for i, ts in enumerate(time_stamps)}
    # dict_index_ts = {i: pd.Timestamp(ts) for i, ts in enumerate(time_stamps)}
    
    # def get_tradetime(x):
    #     index = dict_ts_index[x['tranche']] + x['age']
    #     if index in dict_index_ts.keys() and (dict_index_ts[dict_ts_index[x['tranche']] + x['age']] < last_ts[x['securityid']]):
    #         return dict_index_ts[dict_ts_index[x['tranche']] + x['age']]
    #     else:
    #         return np.nan
    # pos['tradetime'] = pos.apply(get_tradetime, axis=1)
    # pos = pos[(pos['tradetime'].notna())]
    
    # Method Two: which is faster than Method One
    pos['tradetime'] = pos['age'] + pos['tranche']
    pos = pos.drop(pos[pos['tradetime'] > last_ts].index) 
       
    pos = pos.merge(ret_df, on=['tradetime', 'securityid'], how='left')
    pos['new_ret'] = pos['ret'] + 1
    pos['cumret'] = pos.groupby(['securityid', 'tranche'])['new_ret'].cumprod()
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
    port_table['net_value'] = port_table[col_name]
    port_table['ret'] = port_table['net_value'].pct_change(20)
    
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
    factor_data.index = factor_data.index.rename(['tradetime', 'securityid'])
    groups = np.sort(factor_data['factor_quantile'].unique())
    metric_summary = None
    netvalue_summary = None
    
    factor_data['group_size'] = factor_data.groupby(['tradetime', 'factor_quantile'])['factor'].transform(lambda x: len(x))
    factor_data['wt'] = 1 / factor_data['group_size'] / holding_time
    # 分组回测
    for group_i in groups:
        col_name = 'group_'+str(group_i)
        
        portforlio_weight = form_portforlio_weight(factor_data, group_i, max(groups), holding_time)
        stock_pnl = calc_stock_pnl(portforlio_weight, td_ret, holding_time)
        portforlio_pnl = calc_portfolio_pnl(stock_pnl, col_name)
        # from pnl to netvalue
        portforlio_pnl[col_name] = 1+portforlio_pnl[col_name].cumsum()
        if netvalue_summary is None:
            netvalue_summary = portforlio_pnl
        else:
            netvalue_summary = netvalue_summary.merge(portforlio_pnl, how='outer', on='tradetime')
        metric_info = evaluate_pnl(portforlio_pnl, col_name)
        metric_summary = pd.concat([metric_summary, metric_info])
    
    # 多空组回测
    if (long_short):
        col_name = 'hedge'
        portforlio_weight = form_portforlio_hedge_weight(factor_data, max(groups), holding_time)
        stock_pnl = calc_stock_pnl(portforlio_weight, td_ret, holding_time)
        portforlio_pnl = calc_portfolio_pnl(stock_pnl, col_name)
        # from pnl to netvalue
        portforlio_pnl[col_name] = 1+portforlio_pnl[col_name].cumsum()
        netvalue_summary = netvalue_summary.merge(portforlio_pnl, how='outer', on='tradetime')
        metric_info = evaluate_pnl(portforlio_pnl, col_name)
        metric_summary = pd.concat([metric_summary, metric_info])
        
    return metric_summary, netvalue_summary
    

def get_factor_ic_summary_info(ic_data, by_group=False):
    if by_group:
        grouper = ic_data.index.get_level_values('factor_quantile')
        
        ic_mean = ic_data.groupby(grouper).apply(lambda x: x.mean())
        ic_std = ic_data.groupby(grouper).apply(lambda x: x.std())
        icir = ic_mean/ic_std
        ic_summary_table = pd.concat([ic_mean, ic_std, icir], keys=['IC Mean', 'IC Std.', 'ICIR'])
        names = list(ic_summary_table.index.names)
        names[0] = 'Type'
        ic_summary_table.index = ic_summary_table.index.rename(names)
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

def plot_quantile_info_bar(ic_by_q,
                         by_group=False,
                         ylim_percentiles=None,
                         ax=None):
    mean_ret_by_q = ic_by_q.copy()
    DECIMAL_TO_BPS = 1

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(mean_ret_by_q.values,
                                 ylim_percentiles[0]) * DECIMAL_TO_BPS)
        ymax = (np.nanpercentile(mean_ret_by_q.values,
                                 ylim_percentiles[1]) * DECIMAL_TO_BPS)
    else:
        ymin = None
        ymax = None

    if by_group:
        num_group = len(
            mean_ret_by_q.index.get_level_values('group').unique())

        if ax is None:
            v_spaces = ((num_group - 1) // 2) + 1
            f, ax = plt.subplots(v_spaces, 2, sharex=False,
                                 sharey=True, figsize=(18, 6 * v_spaces))
            ax = ax.flatten()

        for a, (sc, cor) in zip(ax, mean_ret_by_q.groupby(level='group')):
            (cor.xs(sc, level='group')
                .multiply(DECIMAL_TO_BPS)
                .plot(kind='bar', title=sc, ax=a))

            a.set(xlabel='', ylabel='Mean Return (bps)',
                  ylim=(ymin, ymax))

        if num_group < len(ax):
            ax[-1].set_visible(False)

        return ax

    else:
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(18, 6))

        (mean_ret_by_q.multiply(DECIMAL_TO_BPS)
            .plot(kind='bar',
                  title="IC Mean By Factor Quantile", ax=ax))
        ax.set(xlabel='', ylabel='IC Mean',
               ylim=(ymin, ymax))

        return ax
    
def plot_quantile_info(quantile_summary, plot_filepath):
    fig, ax = plt.subplots()
    plot_quantile_info_bar(quantile_summary,
                            by_group=False,
                            ylim_percentiles=None,
                            ax=ax)
    fig.savefig(plot_filepath)

def plot_quantile_netvalue(nv_df, plot_filepath):
    # set tradetime as index
    nv_df.set_index('tradetime', inplace=True)

    # plot net value curve for each group
    plt.figure(figsize=(10, 6))
    for col in nv_df.columns:
        plt.plot(nv_df.index, nv_df[col], label=col)

    plt.legend()
    plt.title('Net Value Curve for each groups')
    plt.xlabel('Tradetime')
    plt.ylabel('Net Value')
    plt.savefig(plot_filepath, dpi=300)



def factor_information_coefficient(factor_data, by_group=False):
    def src_ic(group):
        f = group['factor']
        _ic = group[get_forward_returns_columns(factor_data.columns)] \
            .apply(lambda x: stats.spearmanr(x, f)[0])
        return _ic
    
    grouper = [factor_data.index.get_level_values('tradetime')]
    
    if by_group:
        grouper.append('factor_quantile')
    
    ic = factor_data.groupby(grouper).apply(src_ic)
    return ic

def factor_timeSeries_information_coefficient(factor_data):
    def src_ic(group):
        f = group['factor']
        _ic = group[get_forward_returns_columns(factor_data.columns)] \
            .apply(lambda x: stats.spearmanr(x, f)[0])
        return _ic
        
    ic = factor_data.groupby('securityid').apply(src_ic)
    return ic


def factor_group_rough_return(factor_data, dict_col_holding_time = {'1m': 20, '3m': 60, '5m': 100}, long_short=True):
    grouper = ['factor_quantile', 'tradetime']
    def src_avg_return(group):
        return group[get_forward_returns_columns(factor_data.columns)].mean()
    
    avg_return_perTime = factor_data.groupby(grouper).apply(src_avg_return)

    def src_port_rough_return(group):
        total_num = len(group)
        
        values = []
        indices = []
        for col, holding_time in dict_col_holding_time.items():
        
            final_price = 0
            for i in range(holding_time):
                index_list = range(i, total_num, holding_time)
                final_price +=(group.iloc[index_list][col]+1).cumprod().iloc[-1]
            final_price = final_price / holding_time
            values.append(final_price)
            indices.append(col)
        return pd.Series(values, index=indices)
    
    port_rough_return = avg_return_perTime.groupby('factor_quantile').apply(src_port_rough_return)
    port_rough_return = port_rough_return - 1
    if long_short:
        port_rough_return.loc['hedge'] = port_rough_return.loc[1] - port_rough_return.loc[5]
    return port_rough_return
import warnings
import alphalens
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ConstantInputWarning
import sys
sys.path.insert(0, "../")
from alphalens import performance as perf
import dolphindb as ddb 
from alphalens.utils import timedelta_to_string, diff_custom_calendar_timedeltas, infer_trading_calendar, rate_of_return
from alphalens.performance import mean_return_by_quantile
from alphalens.plotting import plot_quantile_returns_bar, plot_quantile_statistics_table
from scipy.stats import mode
import os
import yaml
import matplotlib.pyplot as plt


from factor_cal.config_loader import basic_config as cfg
from factor_cal.table.ddb_table import PriceTable, SecLevelFacTable
from factor_cal.utils import ddb_utils as du


# igmore the warning of ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)
# obtain the ddb session
s = du.DDBSessionSingleton().session

def get_clean_data(factor_name, date, config):
    fac_tb = SecLevelFacTable(config['factor_dbPath'], config['factor_tbName'])
    fac = fac_tb.load_factor(factor_name, date, config['start_time'], config['end_time'], sec_list=None)
    
    price_info = config['price_info']
    pc_tb = PriceTable(price_info['price_dbPath'], price_info['price_tbName'], price_info['time_col'], price_info['sec_col'], price_info['price_cols'])
    price = pc_tb.load_price(date, config['start_time'], config['end_time'], sec_list=None)
    
    fac_df = s.loadTable(tableName=fac).toDF()
    if (fac_df.empty):
        return None
    price_df = s.loadTable(tableName=price).toDF()
    
    # filter the dataframe
    start_time = pd.to_datetime('9:45').time()
    end_time = pd.to_datetime('14:45').time()
    price_df = price_df.set_index('tradetime').between_time(start_time, end_time).reset_index()
    fac_df = fac_df.set_index('tradetime').between_time(start_time, end_time).reset_index()
    
    
    fac_df.set_index(['tradetime', 'securityid'], inplace=True)
    fac_df = fac_df['value']
    fac_df = fac_df.sort_index(level=0)

    price_df = price_df.set_index(['tradetime', 'securityid'])
    prices = price_df['close'].unstack()
    
    
    data=alphalens.utils.get_clean_factor_and_forward_returns(
        fac_df, prices, quantiles=config['quantile'], periods=(20, 60, 100), max_loss=0.5)
    return data


def get_clean_data_OB_avgprice(factor_name, date, config):
    fac_tb = SecLevelFacTable(config['factor_dbPath'], config['factor_tbName'])
    fac = fac_tb.load_factor(factor_name, date, config['start_time'], config['end_time'], sec_list=None)
    
    price_info = config['snap_price_info']
    pc_tb = PriceTable(price_info['price_dbPath'], price_info['price_tbName'], price_info['time_col'], price_info['sec_col'], price_info['price_cols'])
    price = pc_tb.load_price(date, config['start_time'], config['end_time'], sec_list=None)
    
    fac_df = s.loadTable(tableName=fac).toDF()
    if (fac_df.empty):
        return None
    price_df = s.loadTable(tableName=price).toDF()
    
    # filter the dataframe
    start_time = pd.to_datetime('9:30').time()
    end_time = pd.to_datetime('14:57').time()
    price_df = price_df.set_index('tradetime').between_time(start_time, end_time).reset_index()
    fac_df = fac_df.set_index('tradetime').between_time(start_time, end_time).reset_index()
    
    
    fac_df.set_index(['tradetime', 'securityid'], inplace=True)
    fac_df = fac_df['value']
    fac_df = fac_df.sort_index(level=0)

    price_df = price_df.set_index(['tradetime', 'securityid'])
    prices = price_df['bs_avg_price'].unstack()
    
    
    data=alphalens.utils.get_clean_factor_and_forward_returns(
        fac_df, prices, quantiles=config['quantile'], periods=(20, 60, 100), max_loss=0.5)
    return data

def get_clean_data_OB_ask_bid_price(factor_name, date, config):
    fac_tb = SecLevelFacTable(config['factor_dbPath'], config['factor_tbName'])
    fac = fac_tb.load_factor(factor_name, date, config['start_time'], config['end_time'], sec_list=None)
    
    price_info = config['snap_price_info']
    pc_tb = PriceTable(price_info['price_dbPath'], price_info['price_tbName'], price_info['time_col'], price_info['sec_col'], price_info['price_cols'])
    price = pc_tb.load_price(date, config['start_time'], config['end_time'], sec_list=None)
    
    fac_df = s.loadTable(tableName=fac).toDF()
    if (fac_df.empty):
        return None
    price_df = s.loadTable(tableName=price).toDF()
    
    # filter the dataframe
    start_time = pd.to_datetime('9:30').time()
    end_time = pd.to_datetime('14:57').time()
    price_df = price_df.set_index('tradetime').between_time(start_time, end_time).reset_index()
    fac_df = fac_df.set_index('tradetime').between_time(start_time, end_time).reset_index()
    
    
    fac_df.set_index(['tradetime', 'securityid'], inplace=True)
    fac_df = fac_df['value']
    fac_df = fac_df.sort_index(level=0)

    price_df = price_df.set_index(['tradetime', 'securityid'])
    bid1 = price_df['b1'].unstack()
    ask1 = price_df['s1'].unstack()
    
    forward_returns = my_compute_forward_returns(fac_df, bid1, ask1, periods=(20, 60, 100), filter_zscore=20, cumulative_returns=True)
    
    factor_data = alphalens.utils.get_clean_factor(fac_df, forward_returns, groupby=None, 
                                                   groupby_labels=None,
                                                   quantiles=config['quantile'], bins=None, 
                                                   binning_by_group=False, max_loss=0.50, zero_aware=False)
    return factor_data

def my_compute_forward_returns(factor, bid1, ask1, periods=(1, 5, 10), filter_zscore=None, cumulative_returns=True):
    
    factor_dateindex = factor.index.levels[0]
    freq = infer_trading_calendar(factor_dateindex, bid1.index)
    factor_dateindex = factor_dateindex.intersection(bid1.index)
    
    bid1 = bid1.filter(items=factor.index.levels[1])
    ask1 = ask1.filter(items=factor.index.levels[1])
    
    raw_values_dict = {}
    column_list = []
    for period in sorted(periods):
        if cumulative_returns:
            returns = bid1 / ask1.shift(period) - 1
        else:
            returns = bid1 / ask1.shift() - 1

        forward_returns = \
            returns.shift(-period).reindex(factor_dateindex)

        if filter_zscore is not None:
            mask = abs(
                forward_returns - forward_returns.mean()
            ) > (filter_zscore * forward_returns.std())
            forward_returns[mask] = np.nan

        #
        # Find the period length, which will be the column name. We'll test
        # several entries in order to find out the correct period length as
        # there could be non-trading days which would make the computation
        # wrong if made only one test
        #
        entries_to_test = min(
            30,
            len(forward_returns.index),
            len(bid1.index) - period
        )

        days_diffs = []
        for i in range(entries_to_test):
            p_idx = bid1.index.get_loc(forward_returns.index[i])
            start = bid1.index[p_idx]
            end = bid1.index[p_idx + period]
            period_len = diff_custom_calendar_timedeltas(start, end, freq)
            days_diffs.append(period_len.components.days)

        delta_days = period_len.components.days - mode(days_diffs, keepdims=True).mode[0]
        period_len -= pd.Timedelta(days=delta_days)
        label = timedelta_to_string(period_len)

        column_list.append(label)

        raw_values_dict[label] = np.concatenate(forward_returns.values)

    df = pd.DataFrame.from_dict(raw_values_dict)
    df.set_index(
        pd.MultiIndex.from_product(
            [factor_dateindex, bid1.columns],
            names=['date', 'asset']
        ),
        inplace=True
    )
    df = df.reindex(factor.index)

    # now set the columns correctly
    df = df[column_list]

    # df.index.levels[0].freq = freq
    df.index.set_names(['date', 'asset'], inplace=True)
    return df


def get_factor_ic_summary_info(data):
    group_neutral = False
    ic_data = perf.factor_information_coefficient(data, group_neutral)


    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["Risk-Adjusted IC"] = \
        ic_data.mean() / ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0, nan_policy='omit')
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data, nan_policy='omit')
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data, nan_policy='omit')
    ic_summary_table['IC win rate'] = (ic_data > 0).sum() / ic_data.count()
    return ic_summary_table


def get_clean_data_by_type(factor_name, date, config, proc_type):
    if proc_type == 'OB_price':
        return get_clean_data_OB_avgprice(factor_name, date, config)
    elif proc_type == 'bid_ask_price':
        return get_clean_data_OB_ask_bid_price(factor_name, date, config)
    else:
        return get_clean_data(factor_name, date, config)

def do_evaluate(factor_name, proc_type=''):
    date1 = '2023.09.22'
    date2 = '2023.09.25'
    # date2 = '2024.02.20'

    datas = []
    
    for i in pd.date_range(date1, date2):
        data = get_clean_data_by_type(factor_name, i.strftime('%Y.%m.%d'), config, _type)
        if data is not None:
            datas.append(data)

    data = pd.concat(datas, axis=0)

    # ic_summary_table = get_factor_ic_summary_info(data)
    save_path = f'/home/wangzirui/workspace/factor_ic_summary/factor_comb_top_n/{_type}'
    # os.makedirs(save_path, exist_ok=True)
    # ic_summary_table.to_csv(f'{save_path}/{factor_name}.csv')
    
    os.makedirs(f'{save_path}/plots', exist_ok=True)
    plot_filepath = f'{save_path}/plots/{factor_name}_quantile_return.png'
    get_quantile_returns_bar_plot(data, plot_filepath)


def plot_quantile_ic_bar(ic_by_q,
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
    

def get_quantile_returns_bar_plot(factor_data, plot_filepath):
    factor_data = factor_data.rename(columns={'factor_quantile':'group'})
    ic_result_bydate = perf.factor_information_coefficient(factor_data, by_group=True)
    
    ic_quantile_summary = ic_result_bydate.groupby(ic_result_bydate.index.get_level_values('group')).agg('mean')
    
    fig, ax = plt.subplots()
    plot_quantile_ic_bar(ic_quantile_summary,
                            by_group=False,
                            ylim_percentiles=None,
                            ax=ax)
    
    fig.savefig(plot_filepath)


# read config file
config = cfg.BasicConfig('config/config.yml')
# factor_name = "pv_corr"


# proc_types = ['OB_price', 'bid_ask_price', 'close']
proc_types = ['bid_ask_price']
# proc_types = ['close']
pred_type='1m'

base_dir = r'/home/wangzirui/workspace/factor_ic_summary/factor_comb_top_n/bid_ask_price'
factor_filepath = os.path.join(base_dir, f'satisfied_factors_{pred_type}_without_hcorr.yml')
with open(factor_filepath, 'r') as f:
    factor_names = yaml.load(f, Loader=yaml.FullLoader)

for _type in proc_types:
    print("Processing type: ", _type, " ...")
    for facName in factor_names:
        print("Evaluating factor: ", facName, " ...", flush=True)
        do_evaluate(facName, proc_type=_type)

s.close()
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
from alphalens.utils import timedelta_to_string, diff_custom_calendar_timedeltas, infer_trading_calendar
from scipy.stats import mode
import os


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
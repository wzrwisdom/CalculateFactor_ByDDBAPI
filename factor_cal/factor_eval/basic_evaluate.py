from alphalens import performance as perf
from scipy import stats
import pandas as pd

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

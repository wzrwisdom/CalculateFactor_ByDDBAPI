import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from alphalens import performance as perf
from scipy import stats

def preprocess(df):
    df.dropna(inplace=True)
    df= df.replace([np.inf, -np.inf], np.nan).dropna(subset=df.columns)
    return df

test_date = '2024.02.20'
df_tt = pd.read_pickle(f'/data2/prepared_data/fac_ret_{test_date}.pkl')
df_tt = preprocess(df_tt)
X_test, y_test = df_tt.drop(columns=df_tt.columns[:5]), df_tt[['1m']]

scaler = StandardScaler()
X_testScaled = scaler.fit_transform(X_test)
X_testScaled = pd.DataFrame(X_testScaled, columns=X_test.columns, index=X_test.index)
dtest_reg = xgb.DMatrix(X_testScaled, y_test)

model = xgb.Booster()
model.load_model('/home/wangzirui/workspace/models/preliminary_model_with_top_n_factor.json')


preds = model.predict(dtest_reg)

df_test = df_tt.loc[y_test.index.to_list(), df_tt.columns[:5]]
df_test['factor'] = preds
df_test.set_index(['tradetime', 'securityid'], inplace=True)
df_test.index.set_names(['date', 'asset'], inplace=True)



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

ic_summary_table = get_factor_ic_summary_info(df_test)
save_path = '/home/wangzirui/workspace/factor_ic_summary'
ic_summary_table.to_csv(f'{save_path}/factor_comb_top_n.csv')

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from alphalens import performance as perf
from scipy import stats
import os
import yaml

def preprocess(df):
    df.dropna(inplace=True)
    df= df.replace([np.inf, -np.inf], np.nan).dropna(subset=df.columns)
    return df

start_date = '2023.09.22'
end_date = '2023.09.25'

total_df = None
for date in pd.date_range(start_date, end_date):
    date = date.strftime('%Y.%m.%d')
    pickle_filepath = f'/home/wangzirui/workspace/data/fac_ret_{date}.pkl'
    if os.path.exists(pickle_filepath):
        print("processing date: ", date, "...")
        df = pd.read_pickle(pickle_filepath)
        df = preprocess(df)
        total_df = pd.concat([total_df, df], axis=0)

X_test, y_test = total_df.drop(columns=total_df.columns[:5]), total_df[['1m']]

# scaler = StandardScaler()
# X_testScaled = scaler.fit_transform(X_test)
# X_testScaled = pd.DataFrame(X_testScaled, columns=X_test.columns, index=X_test.index)
# dtest_reg = xgb.DMatrix(X_testScaled, y_test)
pred_type='1m'
base_dir = r'/home/wangzirui/workspace/factor_ic_summary/factor_comb_top_n/bid_ask_price'
factor_filepath = os.path.join(base_dir, f'satisfied_factors_{pred_type}_without_hcorr.yml')
with open(factor_filepath, 'r') as f:
    picked_factor_names = yaml.load(f, Loader=yaml.FullLoader)
X_test = X_test[picked_factor_names]
dtest_reg = xgb.DMatrix(X_test, y_test)

model = xgb.Booster()
base_dir = r'/home/wangzirui/workspace/factor_ic_summary/factor_comb_top_n/bid_ask_price'
model.load_model(f'{base_dir}/xgboost_model_no_hcorr.json')


preds = model.predict(dtest_reg)

# df_test = total_df.loc[y_test.index.to_list(), total_df.columns[:5]]
df_test = total_df[total_df.columns[:5]]
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
ic_summary_table.to_csv(f'{base_dir}/factor_comb_top_n.csv')

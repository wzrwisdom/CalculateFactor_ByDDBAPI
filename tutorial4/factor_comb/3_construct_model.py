import pandas as pd
import os
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import yaml


start_date = '2023.09.22'
end_date = '2023.09.22'
# end_date = '2023.09.25'
dates = pd.date_range(start_date, end_date)


def preprocess(df):
    df= df.replace([np.inf, -np.inf], np.nan).dropna(subset=df.columns)
    return df

pred_type='1m'
base_dir = r'/home/wangzirui/workspace/factor_eval_summary/param_optimized'
factor_filepath = os.path.join(base_dir, f'satisfied_factors_nohl.yml')
with open(factor_filepath, 'r') as f:
    picked_factor_names = yaml.load(f, Loader=yaml.FullLoader)

total_df = pd.DataFrame()
for date in dates:
    date = date.strftime('%Y.%m.%d')
    fac_filepath = f'/home/wangzirui/workspace/data/param_optimized/fac_{date}.pkl'
    ret_filepath = f'/home/wangzirui/workspace/data/bid_ask_return/{date}.pkl'
    if os.path.exists(fac_filepath) and os.path.exists(ret_filepath):
        print("processing date: ", date, "...")

        ret_df = pd.read_pickle(ret_filepath)
        fac_df = pd.read_pickle(fac_filepath)
        
        cur_df = ret_df.merge(fac_df, how='right', on=['tradetime', 'securityid'], sort=True)
        cur_df= preprocess(cur_df)
        total_df = pd.concat([total_df, cur_df])
        
X, y = total_df.drop(columns=total_df.columns[:5]), total_df[['1m']]
X = X[picked_factor_names]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)

# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train)
dvalid_reg = xgb.DMatrix(X_valid, y_valid)

# test_date = '2024.02.20'
# df_tt = pd.read_pickle(f'/home/wangzirui/workspace/data/fac_ret_{test_date}.pkl')
# df_tt = preprocess(df_tt)
# X_test, y_test = df_tt.drop(columns=df_tt.columns[:5]), df_tt[['1m']]



# Define hyperparameters
params = {"objective": "reg:squarederror", "tree_method": "hist"}

evals = [(dtrain_reg, "train"), (dvalid_reg, "validation")]
n = 1000
print("Begin training...")
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   evals=evals,
   verbose_eval=100,
   early_stopping_rounds=50,
)

model.save_model(f'{base_dir}/model/xgboost_model_no_hcorr.json')
import pandas as pd
import os
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


start_date = '2023.09.22'
end_date = '2024.02.18'
# end_date = '2023.09.22'
dates = pd.date_range(start_date, end_date)


def preprocess(df):
    df= df.replace([np.inf, -np.inf], np.nan).dropna(subset=df.columns)
    return df

total_df = pd.DataFrame()
for date in dates:
    date = date.strftime('%Y.%m.%d')
    pickle_filepath = f'/data2/prepared_data/fac_ret_{date}.pkl'
    if os.path.exists(pickle_filepath):
        print("processing date: ", date, "...")
        cur_df = pd.read_pickle(pickle_filepath)
        cur_df= preprocess(cur_df)
        total_df = pd.concat([total_df, cur_df])
    
X, y = total_df.drop(columns=total_df.columns[:5]), total_df[['1m']]

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
n = 5000
print("Begin training...")
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   evals=evals,
   verbose_eval=100,
   early_stopping_rounds=50,
)

model.save_model('/home/wangzirui/workspace/models/preliminary_model_with_top_n_factor.json')
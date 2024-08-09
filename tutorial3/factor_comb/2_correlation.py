import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import os

def sortFactor_by_tsIC(factor_names):
    base_dir = '/home/wangzirui/workspace/factor_eval_summary/param_optimized'
    xlsx_filepath = f"{base_dir}/ic_summary.xlsx"
    tsIC_info = pd.read_excel(xlsx_filepath)
    
    
    tsIC_info['metric'] = tsIC_info['IC Mean'].abs()
    selected_df = tsIC_info[tsIC_info['factor'].isin(factor_names)]
    return selected_df.sort_values(['metric'], ascending=False)['factor'].to_list()
     

if __name__ == "__main__":
    base_dir = "/home/wangzirui/workspace/data/param_optimized"
    start_date = '2023.09.22'
    end_date = '2023.09.22'
    df = None
    for date in pd.date_range(start_date, end_date):
        date = date.strftime('%Y.%m.%d')
        filepath = f'{base_dir}/fac_{date}.pkl'
        if os.path.exists(filepath):
            print(filepath)
            tmp_df = pd.read_pickle(filepath)
            df = pd.concat([df, tmp_df], axis=0)
    
               
    cols = df.columns[5:]
    
    # sort the sequence of factor by their ts IC mean
    cols = sortFactor_by_tsIC(list(cols))
    corr_df = df[cols].corr()
    plt.figure(figsize=(40, 36))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig(f'/home/wangzirui/workspace/factor_eval_summary/param_optimized/factor_correlation.png')
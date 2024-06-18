import gzip 
import pickle
import gc
import dolphindb.settings as keys
import warnings
    
# Ignore the UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
    

from factor_cal.table.data_table import HFDataTable
from factor_cal.utils.tools import show_memory

def read_pickle(filepath):
    with gzip.open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def load_order_info(table: HFDataTable, filepath):
    data = read_pickle(filepath)
    
    # data['market'] = 101
    # data.loc[data['code'].str.contains('SZ'), 'market'] = 102
    data.rename(columns={'code': 'securityid'}, inplace=True)
    data['seq_num'] = data['order'] 
    data['tradetime'] = data['date']+data['time']
    data['side'] = data['function_code'].apply(ord)
    data['order_type'] = data['order_kind'].apply(ord)
    
    # show_memory("after creating all new columns")
    data = data[['securityid', 'seq_num', 'date', 'tradetime', 'price', 'volume', 'side', 'order_type']]
    data.__DolphinDB_Type__ = {
        'securityid': keys.DT_SYMBOL,
        'seq_num': keys.DT_LONG,
        'date': keys.DT_DATE,
        'tradetime': keys.DT_TIMESTAMP,
        'price': keys.DT_DOUBLE,
        'volume': keys.DT_INT,
        'side': keys.DT_CHAR,
        'order_type': keys.DT_CHAR
    }
    
    # show_memory("picking columns")
    
    table.save(data)
    # show_memory("after save order data")
    del data
    gc.collect()
    # show_memory("after garbage collection")

def load_trade_info(table: HFDataTable, filepath):
    data = read_pickle(filepath)

    data.rename(columns={
        'code': 'securityid',
        'sell_index': 'sell_seq_num',
        'buy_index': 'buy_seq_num',
        }, inplace=True)
    data['seq_num'] = data['index'] 
    data['tradetime'] = data['date']+data['time']
    data['side'] = data['bs_flag'].apply(lambda x: ord(' ') if x == '' else ord(x))
    
    data = data[['securityid', 'seq_num', 'date', 'tradetime', 'trade_price', 'trade_volume', 'side', 'sell_seq_num', 'buy_seq_num']]
    data.__DolphinDB_Type__ = {
        'securityid': keys.DT_SYMBOL,
        'seq_num': keys.DT_LONG,
        'date': keys.DT_DATE,
        'tradetime': keys.DT_TIMESTAMP,
        'trade_price': keys.DT_DOUBLE,
        'trade_volume': keys.DT_INT,
        'side': keys.DT_CHAR,
        'sell_seq_num': keys.DT_LONG,
        'buy_seq_num': keys.DT_LONG,
    }
    
    table.save(data)
    del data
    gc.collect()

def load_snap_info(table: HFDataTable, filepath):
    data = read_pickle(filepath)

    colRename = {
        'code': 'securityid',
        'Last': 'last_price',
        'volume': 'last_volume',
        }
    data.rename(columns=colRename, inplace=True)
    
    data['tradetime'] = data['date']+data['time']
            
    # data['bid_price'] = data[['bid'+str(i+1) for i in range(10)]].apply(list, axis=1)
    # data['bid_volume'] = data[['bid_size'+str(i+1) for i in range(10)]].apply(list, axis=1)
    # data['ask_price'] = data[['ask'+str(i+1) for i in range(10)]].apply(list, axis=1)
    # data['ask_volume'] = data[['ask_size'+str(i+1) for i in range(10)]].apply(list, axis=1)
    
    select_cols = ['securityid', 'date', 'tradetime', 'last_price', 'last_volume']
    DT_DICT = {
        'securityid': keys.DT_SYMBOL,
        'date': keys.DT_DATE,
        'tradetime': keys.DT_TIMESTAMP,
        'last_price': keys.DT_DOUBLE,
        'last_volume': keys.DT_INT,
    }
    for i in range(10):
        for j in ['bid', 'bid_size', 'ask', 'ask_size']:
            select_cols.append(j+str(i+1))
            if ('size' in j):
                DT_DICT[j+str(i+1)] = keys.DT_INT
            else:
                DT_DICT[j+str(i+1)] = keys.DT_DOUBLE
    data = data[select_cols]
    data.__DolphinDB_Type__ = DT_DICT
    
    table.save(data)
    del data
    gc.collect()
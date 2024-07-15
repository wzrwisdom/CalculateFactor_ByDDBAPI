
import os, sys
sys.path.insert(0, "../")
import datetime

from factor_cal.table.data_table import OrderTable, TradeTable, SnapshotTable
from HFDataPickle.load_info import load_order_info, load_trade_info, load_snap_info 
from factor_cal.utils.tools import show_memory

orderTb = OrderTable("dfs://Level2_data", "OrderRaw")
tradeTb = TradeTable("dfs://Level2_data", "TradeRaw")
snapTb = SnapshotTable("dfs://Level2_data", "SnapshotRaw")


orderTb.create()
tradeTb.create()
snapTb.create()

base_path = "/data2/ddb_data"
start_date = "2023.09.21"
end_date = "2024.02.29"
# start_date = "2024.02.01"
# end_date = "2024.02.29"

start_date = datetime.datetime.strptime(start_date, "%Y.%m.%d")
end_date = datetime.datetime.strptime(end_date, "%Y.%m.%d")

cur_date = start_date
while cur_date <= end_date:
    # construct file path
    cur_date_str = cur_date.strftime("%Y%m%d")
    order_filepath = os.path.join(base_path, 'orders_' + cur_date_str + '.pkl')
    trade_filepath = os.path.join(base_path, 'trades_' + cur_date_str + '.pkl')
    snap_filepath = os.path.join(base_path, 'ticker_' + cur_date_str + '.pkl')
    # print(order_filepath, '\n', trade_filepath, '\n', snap_filepath)

    print("[Current date]: ", cur_date_str, flush=True)
    # if os.path.exists(order_filepath):
    #     print("Loading order info")
    #     show_memory("  before load_order_info")
    #     load_order_info(orderTb, order_filepath)
    
    # if os.path.exists(trade_filepath):
    #     print("Loading trade info")
    #     show_memory("  before load_trade_info")
    #     load_trade_info(tradeTb, trade_filepath)
    
    if os.path.exists(snap_filepath):
        print("Loading snap info")
        show_memory("  before load_snap_info")
        load_snap_info(snapTb, snap_filepath)
    
    cur_date += datetime.timedelta(days=1)

print('Finished!')






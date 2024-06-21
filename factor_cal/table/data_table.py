import dolphindb.settings as keys
import numpy as np
import pandas as pd

from factor_cal.table.ddb_table import BasicTable
from factor_cal.utils.ddb_utils import s

class HFDataTable(BasicTable):
    def __init__(self, db_path, tb_name):
        super(HFDataTable, self).__init__(db_path, tb_name)
        
    def save(self, data:pd.DataFrame):
        nrow = data.shape[0]
        batch_size = 5000000  # about 2.5GB per batch
        num = nrow// batch_size
        num += 1 if nrow % batch_size != 0 else 0
        print("There are {} batches to save".format(num))
        for i in range(num):
            start = i * batch_size
            end = min((i+1) * batch_size, nrow)
            print("\tSaving batch {} from {} to {}".format(i, start, end))
            
            temp_data = data.iloc[start:end]
            temp_data.__DolphinDB_Type__ = data.__DolphinDB_Type__
            tb = s.table(data=temp_data)
            self.get_tb().append(tb)
            s.undef(tb.tableName(), "VAR")
            
    

class OrderTable(HFDataTable):
    
    def _create_db(self):
        if self._exist_db():
            self._drop_db()
        
        dates=np.array(pd.date_range(start='20230401', end='20230410'), dtype="datetime64[D]")
        db1 = s.database(partitionType=keys.VALUE, partitions=dates)
        db2 = s.database(partitionType=keys.HASH, partitions=[keys.DT_SYMBOL, 20])
        db = s.database(partitionType=keys.COMPO, partitions=[db1, db2], dbPath=self.db_path, engine="TSDB")

    def _create_tb(self):
        if self._exist_tb():
            self._drop_tb()
        
        s.run("schema_t = table(100:0, \
            `securityid`seq_num`date`tradetime`price`volume`side`order_type, \
            [SYMBOL, LONG, DATE, TIMESTAMP, DOUBLE, INT, CHAR, CHAR])")
        schema_t = s.table(data="schema_t")
        # pt = self.get_db().createPartitionedTable(schema_t, self.tb_name, 
        #         partitionColumns=["date", "securityid"],
        #         sortColumns=['securityid', 'tradetime'], compressMethods={"tradetime":"delta"},
        #         keepDuplicates="ALL")
        pt = self.get_db().createPartitionedTable(schema_t, self.tb_name, 
                partitionColumns=["date", "securityid"],
                sortColumns=['securityid', 'seq_num', 'tradetime'], compressMethods={"tradetime":"delta"},
                keepDuplicates="LAST", sortKeyMappingFunction=["","hashBucket{, 2}"])
        
class TradeTable(HFDataTable):
    
    def _create_db(self):
        if self._exist_db():
            self._drop_db()
        
        dates=np.array(pd.date_range(start='20230401', end='20230410'), dtype="datetime64[D]")
        db1 = s.database(partitionType=keys.VALUE, partitions=dates)
        db2 = s.database(partitionType=keys.HASH, partitions=[keys.DT_SYMBOL, 20])
        db = s.database(partitionType=keys.COMPO, partitions=[db1, db2], dbPath=self.db_path, engine="TSDB")

    def _create_tb(self):
        if self._exist_tb():
            self._drop_tb()
        
        s.run("schema_t = table(100:0, \
            `securityid`seq_num`date`tradetime`trade_price`trade_volume`side`sell_seq_num`buy_seq_num, \
            [SYMBOL, LONG, DATE, TIMESTAMP, DOUBLE, INT, CHAR, LONG, LONG])")
        schema_t = s.table(data="schema_t")
        # pt = self.get_db().createPartitionedTable(schema_t, self.tb_name, 
        #         partitionColumns=["date", "securityid"],
        #         sortColumns=['securityid', 'tradetime'], compressMethods={"tradetime":"delta"},
        #         keepDuplicates="ALL")
        pt = self.get_db().createPartitionedTable(schema_t, self.tb_name, 
                partitionColumns=["date", "securityid"],
                sortColumns=['securityid', 'seq_num', 'tradetime'], compressMethods={"tradetime":"delta"},
                keepDuplicates="LAST", sortKeyMappingFunction=["","hashBucket{, 2}"])
        
class SnapshotTable(HFDataTable):
    
    def _create_db(self):
        if self._exist_db():
            self._drop_db()
        
        dates=np.array(pd.date_range(start='20230401', end='20230410'), dtype="datetime64[D]")
        db1 = s.database(partitionType=keys.VALUE, partitions=dates)
        db2 = s.database(partitionType=keys.HASH, partitions=[keys.DT_SYMBOL, 20])
        db = s.database(partitionType=keys.COMPO, partitions=[db1, db2], dbPath=self.db_path, engine="TSDB")

    def _create_tb(self):
        if self._exist_tb():
            self._drop_tb()
        
        cols_str = "`securityid`date`tradetime`last_price`last_volume"
        colsType_str = "[SYMBOL, DATE, TIMESTAMP, DOUBLE, INT"
        for i in range(10):
            for j in ['bid', 'bid_size', 'ask', 'ask_size']:
                cols_str += "`" + j + str(i+1)
            colsType_str += ", DOUBLE, INT, DOUBLE, INT"
        colsType_str += "]"
        
        s.run("schema_t = table(100:0, " + cols_str + ", " + colsType_str + ")")
        #     `securityid`date`tradetime`last_price`last_volume`bid_price`bid_volume`offer_price`offer_volume, \
        #     [SYMBOL, DATE, TIMESTAMP, DOUBLE, INT, DOUBLE[], INT[], DOUBLE[], INT[]])")
        schema_t = s.table(data="schema_t")
        pt = self.get_db().createPartitionedTable(schema_t, self.tb_name, 
                partitionColumns=["date", "securityid"],
                sortColumns=['securityid', 'tradetime'], compressMethods={"tradetime":"delta"},
                keepDuplicates="LAST")
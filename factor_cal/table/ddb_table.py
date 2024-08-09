import abc
import numpy as np
import pandas as pd
import dolphindb as ddb
import dolphindb.settings as keys

import factor_cal.utils.ddb_utils as du


class BasicTable:
    def __init__(self, db_path, tb_name):
        self.db_path = db_path
        self.tb_name = tb_name
    
    def _exist_db(self):
        return du.DDBSessionSingleton().get_session().existsDatabase(self.db_path)
    
    def _exist_tb(self):
        return du.DDBSessionSingleton().get_session().existsTable(dbUrl=self.db_path, tableName=self.tb_name)
    
    def _drop_db(self):
        du.DDBSessionSingleton().get_session().dropDatabase(self.db_path)
    
    def _drop_tb(self):
        du.DDBSessionSingleton().get_session().dropTable(dbPath=self.db_path, tableName=self.tb_name)
    
    @abc.abstractmethod
    def _create_db(self):
        pass

    @abc.abstractmethod
    def _create_tb(self):
        pass

    def get_db(self) -> ddb.database:
        return du.DDBSessionSingleton().get_session().database(dbPath=self.db_path)
    
    def get_tb(self) -> ddb.table:
        return du.DDBSessionSingleton().get_session().loadTable(dbPath=self.db_path, tableName=self.tb_name)
    
    @abc.abstractmethod
    def save(self, data):
        pass

    def create(self, overwrite_db=False, overwrite_tb=False):
        if (not self._exist_db()) or overwrite_db:
            self._create_db()
    
        if (not self._exist_tb()) or overwrite_tb:
            self._create_tb()
    
    
class FactorTable(BasicTable):
    def __init__(self, db_path, tb_name):
        super(FactorTable, self).__init__(db_path, tb_name)



class SecLevelFacTable(FactorTable):
    def __init__(self, db_path, tb_name):
        super(FactorTable, self).__init__(db_path, tb_name)

    def _create_db(self):
        if self._exist_db():
            self._drop_db()
        s = du.DDBSessionSingleton().get_session()
        datehours = np.array(pd.date_range(start='2023-04-01 00:00:00', end='2023-04-01 12:00:00', freq='H'), dtype="datetime64[h]")
        db1 = s.database(partitionType=keys.VALUE, partitions=datehours)
        db2 = s.database(partitionType=keys.VALUE, partitions=['f1', 'f2'])
        db = s.database(partitionType=keys.COMPO, partitions=[db1, db2], dbPath=self.db_path, engine="TSDB")

    def _create_tb(self):
        if not self._exist_db():
            raise Warning(f"Database {self.db_path} does not exist")
        if self._exist_tb():
            self._drop_tb()
        s = du.DDBSessionSingleton().get_session()
        s.run("schema_t = table(100:0, `tradetime`securityid`factorname`value, [TIMESTAMP, SYMBOL, SYMBOL, DOUBLE])")
        schema_t = s.table(data="schema_t")
        pt = self.get_db().createPartitionedTable(schema_t, self.tb_name, 
                partitionColumns=["tradetime", "factorname"], 
                sortColumns=["securityid", "tradetime"], compressMethods={"tradetime":"delta"},
                keepDuplicates="LAST", sortKeyMappingFunction=["hashBucket{,500}"])
        
    def save(self, data:pd.DataFrame):
        s = du.DDBSessionSingleton().get_session()
        tb = s.table(data=data)
        self.get_tb().append(tb)
        
    def load_factor(self, fac_name, date, start_time, end_time, sec_list=None):
        table = self.get_tb()
        cols = ["tradetime", "securityid", "factorname", "value"]
        sql = table.select(cols)
        sql = sql.where(f"factorname = '{fac_name}'")
        if start_time is not None:
            condition = f"timestamp(tradetime) >= timestamp({date} {start_time})"
            sql = sql.where(condition)
        if end_time is not None:
            condition = f"timestamp(tradetime) <= timestamp({date} {end_time})"
            sql = sql.where(condition)
        if sec_list is not None:
            condition = f"securityid in {sec_list}"
            sql = sql.where(condition)
        
        sql_line = sql.sort(["securityid", "tradetime"], ascending=True).showSQL()
        table_name = f"t_{self.tb_name}"
        sql_line = table_name + ' = ' + sql_line
        s = du.DDBSessionSingleton().get_session()
        s.run(sql_line)
        return table_name

class PriceTable(BasicTable):
    def __init__(self, db_path, tb_name, time_col, sec_col, price_cols):
        super().__init__(db_path, tb_name)
        self.time_col = time_col
        self.sec_col = sec_col
        self.other_cols = price_cols
    
    def load_price(self, date, start_time, end_time, sec_list=None):
        table = self.get_tb()   
        cols = [self.time_col, self.sec_col] + self.other_cols
        
        sql = table.select(cols)
        if start_time is not None:
            condition = f"timestamp({self.time_col}) >= timestamp({date} {start_time})"
            sql = sql.where(condition)
        if end_time is not None:
            condition = f"timestamp({self.time_col}) <= timestamp({date} {end_time})"
            sql = sql.where(condition)
        if sec_list is not None:
            condition = f"{self.sec_col} in {sec_list}"
        
        sql_line = sql.sort([self.sec_col, self.time_col], ascending=True).showSQL()
        table_name = f"t_{self.tb_name}"
        sql_line = table_name + ' = ' + sql_line
        s = du.DDBSessionSingleton().get_session()
        s.run(sql_line)
        return table_name
    
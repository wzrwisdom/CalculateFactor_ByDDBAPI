import abc
import numpy as np
import pandas as pd
import dolphindb as ddb
import dolphindb.settings as keys

from factor_cal.utils.ddb_utils import s


class BasicTable:
    def __init__(self, db_path, tb_name):
        self.db_path = db_path
        self.tb_name = tb_name
    
    def _exist_db(self):
        return s.existsDatabase(self.db_path)
    
    def _exist_tb(self):
        return s.existsTable(dbUrl=self.db_path, tableName=self.tb_name)
    
    @abc.abstractmethod
    def _create_db(self):
        pass

    @abc.abstractmethod
    def _create_tb(self):
        pass

    @abc.abstractmethod
    def get_db(self):
        pass

    @abc.abstractmethod
    def get_tb(self):
        pass

    def create(self, overwrite_db=False, overwrite_tb=False):
        if (not self._exist_db()) or overwrite_db:
            self._create_db()
    
        if (not self._exist_tb()) or overwrite_tb:
            self._create_tb()
    
class FactorTable(BasicTable):
    def __init__(self, db_path, tb_name):
        super(FactorTable, self).__init__(db_path, tb_name)

    @abc.abstractmethod
    def save(self, data):
        pass

class SecLevelFacTable(FactorTable):
    def __init__(self, db_path, tb_name):
        super(FactorTable, self).__init__(db_path, tb_name)

    def get_db(self):
        return s.database(dbPath=self.db_path)
    
    def get_tb(self) -> ddb.table:
        return s.loadTable(dbPath=self.db_path, tableName=self.tb_name)
    
    def _create_db(self):
        if self._exist_db():
            s.dropDatabase(self.db_path)
        datehours = np.array(pd.date_range(start='2023-04-01 00:00:00', end='2023-04-01 12:00:00', freq='H'), dtype="datetime64[h]")
        db1 = s.database(partitionType=keys.VALUE, partitions=datehours)
        db2 = s.database(partitionType=keys.VALUE, partitions=['f1', 'f2'])
        db = s.database(partitionType=keys.COMPO, partitions=[db1, db2], dbPath=self.db_path, engine="TSDB")

    def _create_tb(self):
        if not self._exist_db():
            raise Warning(f"Database {self.db_path} does not exist")
        if self._exist_tb():
            s.dropTable(dbPath=self.db_path, tableName=self.tb_name)
        
        s.run("schema_t = table(100:0, `tradetime`securityid`factorname`value, [TIMESTAMP, SYMBOL, SYMBOL, DOUBLE])")
        schema_t = s.table(data="schema_t")
        pt = self.get_db().createPartitionedTable(schema_t, self.tb_name, 
                partitionColumns=["tradetime", "factorname"], 
                sortColumns=["securityid", "tradetime"], compressMethods={"tradetime":"delta"},
                keepDuplicates="LAST", sortKeyMappingFunction=["hashBucket{,500}"])
        
    def save(self, data:pd.DataFrame):
        tb = s.table(data=data)
        self.get_tb().append(tb)

        
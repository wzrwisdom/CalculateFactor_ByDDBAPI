import functools
from factor_cal.utils import ddb_utils as du

# Obtain the session object from the singleton instance
s = du.DDBSessionSingleton().get_session()

class DDB_FeatTable:
    def __init__(self, ddb_name, tb_name, time_col, sec_col, other_cols):
        self.ddb_name = ddb_name
        self.tb_name = tb_name
        self.time_col = time_col
        self.sec_col = sec_col
        self.other_cols = other_cols
    
    @functools.lru_cache(maxsize=2)
    def get_table(self, date, start_time=None, end_time=None, sec_list=None):
        table = s.loadTable(dbPath=self.ddb_name, tableName=self.tb_name)
        
        # filter the table by start_time, end_time, and sec_list
        
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
            sql = sql.where(condition)
        
        sql_line = sql.sort([self.sec_col, self.time_col], ascending=True).showSQL()

        # upload the table to the DolphinDB server
        # Todo: we can replace the table name with a hash value
        table_name = f"t_{self.tb_name}"
        sql_line = table_name + ' = ' + sql_line
        s.run(sql_line)
        return table_name
    
    def get_feature(self, feat_colname, date, start_time=None, end_time=None, sec_list=None):
        cur_table_name = self.get_table(date, start_time, end_time, sec_list)
        cur_tb = s.loadTable(tableName=cur_table_name)
        if cur_tb.rows == 0:
            return None
        data = cur_tb.exec(feat_colname).pivotby(self.time_col, self.sec_col).toDF()
        return data
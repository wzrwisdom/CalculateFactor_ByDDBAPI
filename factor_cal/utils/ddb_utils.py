import dolphindb as ddb

class DDBSessionSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.session = ddb.session()
        return cls._instance

    def connect(self, host, port, username, password):
        self.session.connect(host, port, username, password)

    def close(self):
        self.session.close()

    def get_session(self) -> ddb.session:
        return self.session

# Create an instance of DDBSessionSingleton
ddb_session = DDBSessionSingleton()
ddb_session.connect("127.0.0.1", 11282, "wangzr", "wzr123456")

# Obtain the session object from the singleton instance
s = DDBSessionSingleton().get_session()
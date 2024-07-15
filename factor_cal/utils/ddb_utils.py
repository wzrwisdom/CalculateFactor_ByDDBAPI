import dolphindb as ddb
import dolphindb.settings as keys

class DDBSessionSingleton:
    _instance = None

    def __new__(cls, compress=False, protocol=keys.PROTOCOL_DEFAULT):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.session = ddb.session(compress=compress, protocol=protocol)
        return cls._instance

    def connect(self, host, port, username, password):
        self.session.connect(host, port, username, password)

    def close(self):
        self.session.close()

    def get_session(self) -> ddb.session:
        return self.session

# Create an instance of DDBSessionSingleton
ddb_session = DDBSessionSingleton(compress=True, protocol=keys.PROTOCOL_DDB)
# ddb_session.connect("127.0.0.1", 8902, "admin", "123456")
ddb_session.connect("127.0.0.1", 11282, "admin", "123456")

# Obtain the session object from the singleton instance
s = ddb_session.get_session()
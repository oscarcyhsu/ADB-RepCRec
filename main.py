from enum import Enum
from collections import namedtuple
from sys import stdin

# class TransactionState(Enum):
#     BLOCKED = 0
#     WAITING = 1

class Transaction():
    def __init__(self, name, RO, blocked, start):
        self.name = name
        self.RO = RO
        self.blocked = blocked
        self.stratTime = start
        self.locks = dict()
        self.modification = dict()

class DataManager():
    def __init__(self, id: int):
        self.d = dict()

    def __getitem__(self, item):
        return self.d[item]

class LockTable():
    def __init__(self):
        self.table = dict()
    
    def getReadLock(self, T: Transaction, x: str) -> str:
        if x in self.table and T in self.lockTable[x]:
            return self.lockTable[x][T]
        else:
            for locktype in self.table[x].values():
                if locktpye == "R":
                    return None
            self.table[x][T] = "R"
            return self.table[x][T]

    def getWriteLock(self, T: Transaction, data: str) -> str:
        if x in self.table and T not in self.lockTable[x]: # other transaction have lock
            return None
        else:
            self.lockTable[x][T] = "W" # might update lock from readlock to writelock here
            return self.lockTable[x][T]


class TransactionManager():
    def __init__(self):
        dataManagerTuple = namedtuple('dataManagerTuple', ['dataManager', 'lastFail', 'lastRecover'])
        self.time = 0
        self.transactions = dict() # Dict[str, Transaction]
        self.lockTable = LockTable() # Dict[variable, Dict[transaction, R/W] ]
        self.opBuffer = []
        self.conflictGraph = dict() # Dict[Transaction, List[Transaction], key - being waited, value - waiting
    
    def tick(self):
        self.time += 1

    def begin(self, transactionName: str):
        assert(transactionName not in self.transaction)
        T = Transaction(params[0], False, False, TM.time)
        self.transactions[transactionName] = T
    
    def beginRO(self, transactionName: str):
        assert(transactionName not in self.transaction)
        T = Transaction(params[0], True, False, TM.time)
        self.transactions[transactionName] = T

    def read(self, transactionName: str, x: str):
        assert(transactionName in self.transaction)
        T = self.transactions[transactionName].RO
        if T.RO:
            self.__readRO(T, x)
        else:
            self.__read(T, x)
        
    def write(transactionName: str, x: str, val: str):
        val = int(val)

    def dump(self):
        pass

    def end(self, transactionName: str):
        pass

    def fail(self, site: int):
        pass
    
    def recover(self, site: int):
        pass

    def __read(self, T: Transaction, data: str,) -> int:
        if not self.__getLock(T, x, "R"):
            self.opBuffer.append((T, x))
            # TODO: update conflict graph
            # TODO: run deadlock detection
        else:
            # TODO: read from one of the availiable site with x_commit_time > site_last_recover_time
    def __readRO(self, T: Transaction, x: str) -> int:
        pass

    def __getLock(self, T: Transaction, x: str, lockType: str) -> bool:
        if lockType == "R":
            aquiredLock = self.lockTable.getReadLock()
        else:
            aquiredLock = self.lockTable.getWriteLock()

        if aquiredLock == None:
            return False
        else:
            T.locks[data] = aquiredLock



    def __checkDeadLock(self) -> bool: 
        pass

if __name__ == "__main__":

    TM = TransactionManager()

    for line in stdin:
        line = line.strip("\n")
        if line == "":
            continue

        command, params = line.split("(")
        # extract paramenters and get rid of leading/trailing spaces from them
        params = [param.strip(" ") for param in params.strip("()").split(",")]
        if command == "begin":
            TM.begin(param[0])
        elif command == "beginRO":
            TM.beginRO(param[0])
        elif command == "R":
            TM.read(param[0], param[1])
        elif command == "W":
            TM.write(param[0], param[1], param[2])
        elif command == "dump":
            TM.dump()
        elif command == "end":
            TM.end(param[0])
        elif command == "fail":
            TM.end(param[0])
        elif command == "recover":
            TM.recover(param[0])
        else:
            raise Exception("Command Not Found")
        TM.tick()
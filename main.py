from __future__ import annotations

from enum import Enum
from collections import namedtuple, defaultdict
from sys import stdin

# class TransactionState(Enum):
#     BLOCKED = 0
#     WAITING = 1

# locktable, conflictgraph maintanance
# at read/write operation: add lock to locktable, add edge to conflictgraph
# at end: clean all T's lock in locktable, clean all edges connected to T on conflictgraph
# how to clean lock in locktable:
# iterate through all lock  and find T's lock (slow)
# iterate through all data items that T has lock on

class Transaction():
    def __init__(self, name, RO, blocked, start):
        self.name = name
        self.RO = RO
        self.blocked = blocked
        self.startTime = start
        self.locks = dict()
        self.variableFinalValues = dict()

DATA_MANAGER_FAIL = "fail"
DATA_MANAGER_RECOVER = "recover"

class DataManager():
    def __init__(self, dataManagerId: int, initTime: int):
        # { 'variableName': [(value, commitTime)] }
        self.variableValues = defaultdict(list)
        for variableIndex in range(1, 21):
            # Each variable xi is initialized to the value 10i (10 times i)
            variableValue = 10 * variableIndex

            if variableIndex % 2 == 1:
                # odd indexed variables are at site 1 + (indexNumber mod 10)
                if dataManagerId == (1 + variableIndex % 10):
                    self.variableValues["x{}".format(variableIndex)].append((variableValue, initTime))
            else:
                # even indexed variables are at all sites
                self.variableValues["x{}".format(variableIndex)].append((variableValue, initTime))


class State(Enum):
    GRANTED = 0
    WAITING = 1

    
class LockTable():

    def __init__(self):
        self.LockTuple = namedtuple('LockTuple', ['T', 'type', 'state'])
        self.table = {"x" + str(i) : [] for i in range(1, 21)} # Dict[variable, List[(transaction, R/W, state)]]

    """
    return a tuple (lock, transaction)
    lock: the lock T get after request a lock - R/W/None
    T': the transation that prevents T from getting a lock - T'/None
    """
    def getReadLock(self, T: Transaction, x: str) -> tuple[str, Transaction]:
        curLock = self.__getExistLock(T, x)
        if curLock:
            if curLock.state == State.GRANTED:
                return curLock.type, None
            else:
                raise Exception("encounter blocked operation from the same transaction")

        if len(self.table[x]) == 0:
            self.table[x].append(self.LockTuple(T, "R", State.GRANTED))
            return "R", None
        elif self.table[x][-1].state == State.GRANTED and self.table[x][-1].type == "R":
            self.table[x].append(self.LockTuple(T, "R", State.GRANTED))
            return "R", None
        else:
            self.table[x].append(self.LockTuple(T, "R", State.WAITING))
            return None, self.table[x][-2].T

    """
    return a tuple (lock, T')
    lock: the lock T get after request a lock - W/None
    T': the transation that prevents T from getting a lock - T'/None
    """
    def getWriteLock(self, T: Transaction, x: str) -> tuple[str, Transaction]:
        curLock = self.__getExistLock(T, x)
        if curLock:
            if curLock.type == "W" and curLock.state == State.GRANTED:
                return curLock.type, None
            else:
                raise Exception("encounter blocked operation from the same transaction")
        

        if len(self.table[x]) == 0:
            newLock = self.LockTuple(T, "W", State.GRANTED)
            self.table[x].append(newLock)
            return newLock.type, None
        else:
            blockingLock = self.table[x][-1]
            self.table[x].append(self.LockTuple(T, "W", State.WAITING))
            return None, blockingLock.T

    '''
    release the lock acquired by T on x. If x is None, release all lock aquired by T.
    '''
    def releaseLock(self, T: Transaction, x:str = None):
        if x != None:
            self.table[x] = list(filter(lambda x:x.T != T, self.table[x]))
            for i in range(len(self.table[x])):
                if i == 0:
                    self.table[x][i].state = State.GRANTED
                elif self.table[x][i-1].type != self.table[x][i].type: # one read-lock and one writelock
                    self.table[x][i].state = State.WAITING
                else:
                    self.table[x][i].state = self.table[x][i-1].state
        else:
            for j in range(1, 21):
                x = "x"+str(j)
                self.table[x] = list(filter(lambda x:x.T != T, self.table[x]))

                for i in range(len(self.table[x])):
                    if i == 0:
                        self.table[x][i] = self.table[x][i]._replace(state = State.GRANTED)
                    elif self.table[x][i-1].type != self.table[x][i].type: # one read-lock and one writelock
                        self.table[x][i] = self.table[x][i]._replace(state = State.WAITING)
                    else:
                        self.table[x][i] = self.table[x][i]._replace(state = self.table[x][i-1].state)
    """
    utility function to return current lock that T has on x. If not exist, return None
    """
    def __getExistLock(self, T: Transaction, x: str): 
        for lock in self.table[x]:
            if lock.T == T:
                return lock
        return None

    '''
    bool : whether the there exists a deadlock
    Transacion: the transation to kill to resolve the deadlock
    '''
    def checkDeadLock(self) -> tuple[bool, Transaction]:
        
        # build graph
        graph = defaultdict(list)
        visited = defaultdict(int)
        for queue in self.table.values():
            # build edge
            for i in range(1, len(queue)):
                if queue[i].state == State.WAITING:
                    graph[queue[i-1].T].append(queue[i].T)
            # get all node
            for T, _, _ in queue:
                visited[T] = 0
        # cycle detection
        def dfs(T: Transaction, trail):
            if visited[T] == 1:
                return True, trail
            elif visited[T] == 2:
                return False, []

            visited[T] = 1
            trail.append(T)
            for neighbor in graph[T]:
                if dfs(neighbor, trail)[0]:
                    return True, trail
            visited[T] = 2
            trail.pop()

            return False, []

        
        for node in visited.keys():
            haveCycle, trail = dfs(node, [])
            if haveCycle:
                youngestT, youngestAge = None, None
                for T in trail:
                    if youngestT == None or T.startTime > youngestAge:
                        youngestT = T
                        youngestAge = T.startTime
                return True, youngestT
        return False, None


class TransactionManager():

    def __init__(self):
        dataManagerTuple = namedtuple('dataManagerTuple', ['dataManager', 'lastFail', 'lastRecover'])
        self.time = 0
        self.transactions = dict() # Dict[str, Transaction]
        self.lockTable = LockTable()
        # instructions waiting because of lock conflict (write, regular read) or site failure (RO read)
        self.instructionBuffer = []
        self.dataManagers = [DataManager(i, self.time) for i in range(1, 11)]

        # { dataManagerIdx: [(DATA_MANAGER_FAIL, time1), (DATA_MANAGER_RECOVER, time2)] }
        self.dataManagerStatusHistory = defaultdict(list)

        # self.conflictGraph = dict() # Dict[Transaction, List[Transaction], key - being waited, value - waiting
    

    # when an instruction in the instructionBuffer becomes unblocked, this function can be reused to run it
    def runInstruction(self, line: str):
        print(f"run Instruction {line}")
        command, params = line.split("(")
        params = [param.strip() for param in params.strip("()").split(",")]

        if command == "begin":
            self.begin(params[0])
        elif command == "beginRO":
            self.beginRO(params[0])
        elif command == "R":
            if not self.read(params[0], params[1]):
                self.instructionBuffer.append(line)
        elif command == "W":
            self.write(params[0], params[1], params[2])
        elif command == "dump":
            self.dump()
        elif command == "end":
            self.end(params[0])
        elif command == "fail":
            self.end(params[0])
        elif command == "recover":
            self.recover(params[0])
        else:
            raise Exception("Command Not Found")

    def tick(self):
        self.time += 1

    def begin(self, transactionName: str):
        assert(transactionName not in self.transactions)
        T = Transaction(transactionName, False, False, TM.time)
        self.transactions[transactionName] = T
    
    def beginRO(self, transactionName: str):
        assert(transactionName not in self.transactions)
        T = Transaction(transactionName, True, False, TM.time)
        self.transactions[transactionName] = T


    # return True/False depending on whether the read was blocked
    # print out the value of the variable if not blocked
    def read(self, transactionName: str, x: str):
        assert(transactionName in self.transactions)
        T = self.transactions[transactionName]
        if T.RO:
            variableIdx = int(x[1:])
            variableIsReplicated = (variableIdx % 2 == 0)
            if variableIsReplicated:
                for dataManagerIdx, dataManager in enumerate(self.dataManagers):
                    # find the most recent value of x that was committed earlier than T’s start time
                    for value, commitTime in dataManager.variableValues[x][::-1]:
                        if commitTime < T.startTime:
                            # if there are no failure timestamps in between x’s commit time and T’s start time, read the value
                            valueIsValid = True
                            for status, timestamp in self.dataManagerStatusHistory[dataManagerIdx]:
                                if status == DATA_MANAGER_FAIL and timestamp > commitTime and timestamp < T.startTime:
                                    valueIsValid = False
                                    break
                            if valueIsValid:
                                print(f"{x}: {value}")
                                return True
                            
                            break # we don't care about earlier committed values of x
                
                # no dataManager has a valid value for x, T should abort
                # TODO: write a test for this
                print(f"No possible RO read for replicated variable available, aborting transaction {T.name}")
                self.abortTransaction(T)
                return False

            # variable is not replicated. If the corresponding site is up, read it, else wait
            dataManagerIdx = variableIdx % 10
            if not self.dataManagerStatusHistory[dataManagerIdx] or \
                self.dataManagerStatusHistory[dataManagerIdx][-1][0] == DATA_MANAGER_RECOVER:

                dataManager = self.dataManagers[dataManagerIdx]
                for value, commitTime in dataManager.variableValues[x][::-1]:
                    if commitTime < T.startTime:
                        print(f"{x}: {value}")
                        return True

            return False

        # regular read
        if self.__getLock(T, x, "R"):
            # Find a version of x on any site that is up with x_commit_time > site_last_recover_time
            for dataManagerIdx, dataManager in enumerate(self.dataManagers):
                if (x not in dataManager.variableValues) or (not self.dataManagerIsUp(dataManagerIdx)):
                    continue
                value, variableCommitTime = dataManager.variableValues[x][-1]
                dataManagerLastRecoverTime = -1
                # we already know the DataManager is alive here; extract its most recent recovery time if it exists
                statusHistory = self.dataManagerStatusHistory[dataManagerIdx]
                if statusHistory:
                    dataManagerLastRecoverTime = statusHistory[-1][1]

                if variableCommitTime > dataManagerLastRecoverTime:
                    print(f"{x}: {value}")
                    return True

        return False


    def dataManagerIsUp(self, dataManagerIdx):
        return (dataManagerIdx not in self.dataManagerStatusHistory) or \
            self.dataManagerStatusHistory[dataManagerIdx][-1][0] == DATA_MANAGER_RECOVER
    

    # return True/False depending on whether the write was blocked
    def write(self, transactionName: str, x: str, val: str):
        assert(transactionName in self.transactions)
        T = self.transactions[transactionName]
        
        aquiredLock = self.__getLock(T, x, "W")

        hasDeadLock, victim = self.lockTable.checkDeadLock()
        while hasDeadLock:
            print(f"Deadlock detected, victim is {victim.name}")
            self.abortTransaction(victim)
            hasDeadLock, victim = self.lockTable.checkDeadLock()
        
        if not aquiredLock:
            return False
        if T.name not in self.transactions: # checkDeadLock abort T
            return False

        T.variableFinalValues[x] = val
        return True


    def abortTransaction(self, T: Transaction):
        print(f"Aborting transaction {T.name}")
        self.lockTable.releaseLock(T)
        del self.transactions[T.name]


    def dump(self):
        for i, dataManager in enumerate(self.dataManagers):
            print(f"==== data in dataManager{i+1} ====")
            print('   '.join(['{}:{}'.format(variable, value_timestamp_list[-1][0])
                for variable, value_timestamp_list in dataManager.variableValues.items()]))

    def end(self, transactionName: str):
        assert(transactionName in self.transactions)
        T = self.transactions[transactionName]
        print(f"Commiting transaction {T.name}")

        affectedDataManagerIndices = set()
        for variable, value in T.variableFinalValues.items():
            variableIdx = int(variable[1:])
            dataManagerIndices = []
            for i in range(1, 11):
                if variableIdx % 2 == 0:
                    dataManagerIndices.append(i - 1)
                elif i == (1 + (variableIdx % 10)): # odd variable index
                    dataManagerIndices.append(i - 1)
            
            livingDataManagerIndices = []
            for i in dataManagerIndices:
                statusHistory = self.dataManagerStatusHistory.get(i)
                if not statusHistory or statusHistory[-1][0] == DATA_MANAGER_RECOVER:
                    # no failure history or recently recovered
                    # TODO: check site didn't fail between start and end of transaction
                    livingDataManagerIndices.append(i)

            for i in livingDataManagerIndices:
                livingDataManager = self.dataManagers[i]
                livingDataManager.variableValues[variable].append((value, self.time))
                affectedDataManagerIndices.add(i+1)
        
        for i in affectedDataManagerIndices:
            print(f"site {i} is written by transaction {T.name}")
        
        self.lockTable.releaseLock(T)
        del self.transactions[transactionName]

    def fail(self, site: int):
        pass
    
    def recover(self, site: int):
        # TODO: scan instructionBuffer to see if any waiting instructions can now be run (if yes, they should be RO reads)
        pass

    
    def __getLock(self, T: Transaction, x: str, lockType: str) -> bool:
        if lockType == "R":
            aquiredLock, blockingTransaction = self.lockTable.getReadLock(T, x)
        else:
            aquiredLock, blockingTransaction = self.lockTable.getWriteLock(T, x)
        
        if not aquiredLock:
            print(f"Transaction {T.name} wait for transaction {blockingTransaction.name} " 
                   "because of lock conflict")
        else:
            T.locks[x] = aquiredLock

        return aquiredLock is not None



    def __resolveDeadLock(self) -> bool: 
        pass

def test_exclusive_lock():
    LT = LockTable()
    T1 = Transaction("T1", False, False, 0)
    T2 = Transaction("T2", False, False, 1)

    LT.getWriteLock(T1, "x1")
    assert(LT.getWriteLock(T2, "x1") == (None, T1))

def test_shared_and_exclusive_lock():
    LT = LockTable()
    T1 = Transaction("T1", False, False, 0)
    T2 = Transaction("T2", False, False, 1)

    LT.getReadLock(T1, "x2")
    assert(LT.getWriteLock(T2, "x2") == (None, T1))

def test_release_lock():
    LT = LockTable()
    T1 = Transaction("T1", False, False, 0)
    T2 = Transaction("T2", False, False, 1)

    LT.getReadLock(T1, "x2")
    LT.releaseLock(T1, "x2")
    assert(LT.getWriteLock(T2, "x2") == ("W", None))

def test_starvation():
    LT = LockTable()
    T1 = Transaction("T1", False, False, 0)
    T2 = Transaction("T2", False, False, 1)
    T3 = Transaction("T3", False, False, 2)

    LT.getReadLock(T1, "x1")
    LT.getWriteLock(T2, "x1")
    assert(LT.getReadLock(T3, "x1") == (None, T2))
    
def test_repeated_lock():
    LT = LockTable()
    T1 = Transaction("T1", False, False, 0)
    T2 = Transaction("T2", False, False, 1)

    LT.getReadLock(T1, "x1")
    assert(LT.getReadLock(T1, "x1") == ("R", None))

    LT.getWriteLock(T1, "x2")
    assert(LT.getReadLock(T1, "x2") == ("W", None))
    assert(LT.getWriteLock(T1, "x2") == ("W", None))

def test_dead_lock():
    LT = LockTable()
    T1 = Transaction("T1", False, False, 0)
    T2 = Transaction("T2", False, False, 1)

    LT.getWriteLock(T1, "x1")
    LT.getWriteLock(T2, "x2")
    LT.getWriteLock(T1, "x2")
    LT.getWriteLock(T2, "x1")
    assert(LT.checkDeadLock() == (True, T2))

# test_exclusive_lock()
# test_shared_and_exclusive_lock()
# test_release_lock()
# test_starvation()
# test_repeated_lock()
# test_dead_lock()

TM = TransactionManager()
for line in stdin:
    line = line.strip()
    if line == "" or line.startswith("//"):
        continue
    TM.runInstruction(line)
    TM.tick()


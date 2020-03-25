import numpy as np

"""
q[(s, a)] object
"""
class doubleDefaultDict:
    def __init__(self):
        self.obj = {}

    def getActions(self, state):
        # if actions have not been recorded before, return empty dict
        if state not in self.obj:
            self.obj[state] = {}
        return self.obj[state]

    def __getitem__(self, stateAction):
        state, action = stateAction

        if state not in self.obj:
            self.obj[state] = {}

        # by default if item doesn't exist, return 0
        if action not in self.obj[state]:
            self.obj[state][action] = np.float("-inf")

        return self.obj[state][action]

    def __setitem__(self, stateAction, value):
        state, action = stateAction

        if state not in self.obj:
            self.obj[state] = {}

        self.obj[state][action] = value

"""
convert state/action array to dictionary key
"""
def array2key(arr):
    reshapeArr = arr.flatten()
    reshapeArr = [str(a) for a in reshapeArr]
    return "_".join(reshapeArr)

"""
convert dictionary key to state/action array
"""
def key2array(key):
    arr = key.split("_")
    arr = np.array([np.float(a) for a in arr])
    return arr

"""
find the maximum (key, value) in dictionary
"""
def getMaxDict(dicts):
    # if dictionary is empty
    if (len(dicts) == 0):
        return (None, np.float("-inf"))

    setKeysMax = []
    maxVal = np.float("-inf")
    for k, v in dicts.items():
        if v == maxVal:
            setKeysMax.append(k)
        elif v > maxVal:
            maxVal = v
            setKeysMax = [k]

    # choose randomly. return (maxState, maxVal)
    return (np.random.choice(setKeysMax), maxVal)
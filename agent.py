import numpy as np
import qlearning

class DemandAgent():
    def __init__(self, muDemand, stdDemand):

        self.muDemand = muDemand
        self.stdDemand = stdDemand

    def genDemand(self):
        demand  = max(0, np.random.normal(self.muDemand, self.stdDemand))
        return demand

"""
Central Planner based on Q-Learning
takes state and action from all agents, make decision on behalf of them
"""
class Planner():
    def __init__(self, learningParams, retailerOrder):
        # create q[(s, a)]
        self.q = qlearning.doubleDefaultDict()

        # learning params
        self.alpha = learningParams["alpha"]
        self.epsilon = learningParams["epsilon"]
        self.gamma = learningParams["gamma"]

        self.retailerOrder = retailerOrder

    def resetAction(self):
        action = np.array([
            [np.nan, np.nan, 0],  # serviceTime
            [0, 0, 0],  # orderToSupplier
            [np.nan, np.nan, np.nan] # reorderPoint
        ])

        return action

    def chooseRandomAction(self, state, demand):
        # initiate action
        action = self.resetAction()

        # state, action of every nodes
        retailerState = state[:, 2]
        retailerAction = action[:, 2]

        # fixed order
        retailerAction[1] = self.retailerOrder
        # try random retailerAction
        # retailerMaxOrder = retailerState[6] - retailerState[0]  # capacity - current inventory
        # retailerInventoryLack = max(0, demand - retailerState[0])  # gap between customer order and inventory
        # retailerAction[1] = np.random.choice(range(int(retailerInventoryLack), retailerMaxOrder))

        s1State = state[:, 1]
        s1Action = action[:, 1]
        s0State = state[:, 0]
        s0Action = action[:, 0]

        # ORDER:
        s1Action[1] = np.random.choice(range(
            int(retailerAction[1] - s1State[0]), int(s1State[6] - s1State[0])
        ))
        s1Action[1] = np.clip(s1Action[1], 0, None)
        s0Action[1] = np.random.choice(range(
            int(s1Action[1] - s0State[0]), int(s0State[6] - s0State[0])
        ))
        s0Action[1] = np.clip(s0Action[1], 0, None)

        # SERVICE TIME : if supplier's inventory > demand, serviceTime = 0
        s0Action[0] = 0 + s0State[5] if s0State[0] < s1Action[1] else 0
        s1Action[0] = s0Action[0] + s1State[5] if s1State[0] < retailerAction[1] else 0

        # choose reorder point for next cycle
        # reorder point up to capacity
        # retailerAction[2] = np.random.choice(range(retailerState[6])) # 0 to max capacity
        # reorder point up to new inventory position
        newInventoryPosition = int(retailerAction[1] + retailerState[0])
        retailerAction[2] = np.random.choice(range(newInventoryPosition))

        return action

    def chooseGreedyAction(self, state, demand):
        # initiate action
        action = self.resetAction()

        # state, action of every nodes
        retailerState = state[:, 2]
        retailerAction = action[:, 2]
        s1State = state[:, 1]
        s1Action = action[:, 1]
        s0State = state[:, 0]
        s0Action = action[:, 0]

        # pick maximum action and convert to array
        # get inventory as state
        s = qlearning.array2key(state[0])
        listActions, _ = qlearning.getMaxDict(self.q.getActions(s))
        # get suppliersOrders
        suppliersOrders = qlearning.key2array(listActions)
        s0Action[1] = suppliersOrders[0]
        s1Action[1] = suppliersOrders[1]
        retailerAction[1] = suppliersOrders[2]

        # get reorderPoint
        retailerAction[2] = suppliersOrders[3]

        # if supplier's inventory > demand, serviceTime = 0
        s0Action[0] = 0 + s0State[5] if s0State[0] < s1Action[1] else 0
        s1Action[0] = s0Action[0] + s1State[5] if s1State[0] < retailerAction[1] else 0
        retailerAction[0] = 0 # always 0 service time

        return action

    def takeAction(self, state, demand):
        # epsilon-greedy
        # explore
        if np.random.uniform() < self.epsilon:
            action = self.chooseRandomAction(state, demand)
        # exploit
        else:
            # choose maximum with random tie braking if multiple maximum
            s = qlearning.array2key(state[0])
            actionsList = self.q.getActions(s)

            # if unknown actions
            if len(actionsList) == 0:
                action = self.chooseRandomAction(state, demand)
            else:
                action = self.chooseGreedyAction(state, demand)

        return action

    def train(self, oldState, oldAction, newState, reward):
        new_s = qlearning.array2key(newState[0])

        # train q for old state and action in the next actionTrigger
        if (oldState is not None) & (oldAction is not None):
            old_s = qlearning.array2key(oldState[0]) # inventory

            # action = [order qty] + [reorderPoint]
            a = np.append(oldAction[1], oldAction[2, 2])
            old_a = qlearning.array2key(a) # order quantity

            actionsList = self.q.getActions(new_s)
            _, maxQ = qlearning.getMaxDict(actionsList)

            maxQ = maxQ if maxQ != np.float("-inf") else 0
            current_q = self.q[(old_s, old_a)] if self.q[(old_s, old_a)] != np.float("-inf") else 0

            self.q[(old_s, old_a)] = current_q + self.alpha * (reward + self.gamma * maxQ - current_q)

        return

# """
# Actor-Critic Policy Gradient
# """
# from scipy.spatial import distance
#
# class ValueEstimator():
#     def __init__(self, α=0.1):
#         self.w = np.zeros(3, )  # state dimension is 3
#         self.α = α
#
#         self.v_dw = np.zeros(3, )
#         self.β = 0.9
#         self.ε = 1e-10
#
#     def predict(self, state):
#         value = self.w @ state
#         return value
#
#     def update(self, state, target):
#         value = self.predict(state)
#
#         # update w
#         dloss = 2 * (value - target) * state
#         # minimize RMSPROP
#         self.v_dw = self.β * self.v_dw + (1 - self.β) * (dloss ** 2)
#         self.w -= self.α * dloss / (np.sqrt(self.v_dw) + self.ε)
#
#
# class PolicyEstimator():
#     def __init__(self, α=0.01):
#         # assume diagonal covariance
#         self.θ0 = np.zeros(3,)
#         self.θ1 = np.zeros(3,)
#         self.θ2 = np.zeros(3,)
#         self.σ = 1
#         self.α = α
#
#         self.losses = []
#
#         self.v_dθ0 = np.zeros(3,)
#         self.v_dθ1 = np.zeros(3,)
#         self.v_dθ2 = np.zeros(3,)
#         self.β = 0.9
#         self.ε = 1e-10
#
#     def mu_state(self, state):
#         mu = self.θ @ state
#
#         return mu
#
#     def predict(self, state):
#         mu = self.mu_state(state)
#
#         action = np.random.multivariate_normal(mu, self.Σ)
#         action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
#
#         return action
#
#     def update(self, state, target, action):
#         mu = self.mu_state(state)
#         action = self.predict(state)
#
#
#         # update θ
#         dloss = -np.outer(action-mu, state) * target
#         # minimize RMSPROP
#         self.v_dθ = self.β * self.v_dθ + (1 - self.β) * (dloss ** 2)
#         # element-wise division by sqrt
#         self.θ -= self.α * dloss / (np.sqrt(self.v_dθ) + self.ε)
#

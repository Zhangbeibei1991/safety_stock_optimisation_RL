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
        s1State = state[:, 1]
        s1Action = action[:, 1]
        s0State = state[:, 0]
        s0Action = action[:, 0]

        # fixed order
        retailerAction[1] = self.retailerOrder

        # choose reorder point for next cycle
        # reorder point up to capacity
        retailerAction[2] = np.random.choice(range(
            int(retailerAction[1] + retailerState[0])
        ))

        # ORDER:
        # action_i+1 - inventory_i <= action_i <= capacity_i - inventory_i
        # action_i >= 0 (clipped at 0)
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

"""
Actor-Critic Policy Gradient
"""
class ValueEstimator():
    def __init__(self, α=0.1):
        self.w = np.zeros(3, )  # state dimension is 3
        self.α = α

        self.v_dw = np.zeros(3, )
        self.β = 0.9
        self.ε = 1e-10

    def predict(self, s):
        value = self.w @ s
        return value

    def update(self, s, target):
        value = self.predict(s)

        # update w
        dloss = 2 * (value - target) * s
        # minimize RMSPROP
        self.v_dw = self.β * self.v_dw + (1 - self.β) * (dloss ** 2)
        self.w -= self.α * dloss / (np.sqrt(self.v_dw) + self.ε)


class PolicyEstimator():
    def __init__(self, α=0.01):
        # assume diagonal covariance
        self.θ = np.zeros((3, 3))
        self.σ = 20
        self.α = α

        self.losses = []

        self.v_dθ = np.zeros((3, 3))
        self.β = 0.9
        self.ε = 1e-10

    def predict(self, s, capacity, retailerOrderQty):
        # mu = self.mu_state(state)

        """
        decide in this order:
        1. reorder point for retailer
        2. order for node 1
        3. order for node 0
        """

        # capacity = state[6]
        # retailerOrderQty = self.retailerOrder
        # s = state[0]  # inventory

        a = [np.nan, np.nan, np.nan]
        mu = [np.nan, np.nan, np.nan]
        # choose reorder point for next cycle
        # reorder point up to capacity
        mu[2] = self.θ[2] @ s
        a[2] = np.random.normal(mu[2], self.σ)
        a[2] = np.clip(a[2], 0, int(retailerOrderQty + s[2]))

        mu[1] = self.θ[1] @ s
        a[1] = np.random.normal(mu[1], self.σ)
        a[1] = np.clip(a[1], retailerOrderQty-s[1], capacity[1]-s[1])
        a[1] = np.clip(a[1], 0, None)  # limit to 0

        mu[0] = self.θ[0] @ s
        a[0] = np.random.normal(mu[0], self.σ)
        a[0] = np.clip(a[0], a[1]-s[0], capacity[0]-s[0])
        a[0] = np.clip(a[0], 0, None)  # limit to 0

        return mu, a

    def update(self, s, target, a, capacity, retailerOrderQty):
        # TODO: old mu must be stored
        # decide parameter
        mu, _ = self.predict(s, capacity, retailerOrderQty)

        # update θ
        dloss = -np.outer(a - mu, s) * target
        # minimize RMSPROP
        self.v_dθ = self.β * self.v_dθ + (1 - self.β) * (dloss ** 2)
        # element-wise division by sqrt
        self.θ -= self.α * dloss / (np.sqrt(self.v_dθ) + self.ε)


class PlannerWithPolicyGradient():
    def __init__(self, learningParams, retailerOrder):
        # create policy and actor
        self.policy_estimator = PolicyEstimator(α=0.001)
        self.value_estimator = ValueEstimator(α=0.1)

        # learning params
        # self.alpha = learningParams["alpha"]
        # self.epsilon = learningParams["epsilon"]
        # self.gamma = learningParams["gamma"]
        self.discount_factor = 0.95

        self.retailerOrder = retailerOrder

    def resetAction(self):
        action = np.array([
            [np.nan, np.nan, 0],  # serviceTime
            [0, 0, 0],  # orderToSupplier
            [np.nan, np.nan, np.nan] # reorderPoint
        ])

        return action

    def takeAction(self, state, demand):

        # state for actor-critic
        s = state[0]
        capacity = state[6]
        _, a = self.policy_estimator.predict(s, capacity, self.retailerOrder)

        # initiate action
        action = self.resetAction()

        # state, action of every nodes
        retailerState = state[:, 2]
        retailerAction = action[:, 2]
        s1State = state[:, 1]
        s1Action = action[:, 1]
        s0State = state[:, 0]
        s0Action = action[:, 0]

        # fixed order
        retailerAction[1] = self.retailerOrder

        # choose reorder point for next cycle
        retailerAction[2] = a[2]
        s1Action[1] = a[1]
        s0Action[1] = a[0]

        # SERVICE TIME : if supplier's inventory > demand, serviceTime = 0
        s0Action[0] = 0 + s0State[5] if s0State[0] < s1Action[1] else 0
        s1Action[0] = s0Action[0] + s1State[5] if s1State[0] < retailerAction[1] else 0

        return action

    def train(self, oldState, oldAction, newState, reward):

        # train for old state and action in the next actionTrigger
        if (oldState is not None) & (oldAction is not None):
            new_s = newState[0]
            old_s = oldState[0] # inventory
            old_a = np.append(oldAction[1, :2], oldAction[2, 2])

            # calculate TD target
            value_now = self.value_estimator.predict(old_s)
            value_next = self.value_estimator.predict(new_s)
            td_target = reward + self.discount_factor * value_next
            td_error = td_target - value_now

            # update the value estimator
            self.value_estimator.update(old_s, td_target)
            # update the policy estimator
            capacity = oldState[6]
            self.policy_estimator.update(old_s, td_error, old_a, capacity, self.retailerOrder)

        return

import numpy as np
import qlearning

class DemandAgent():
    def __init__(self, muDemand, stdDemand):

        self.muDemand = muDemand
        self.stdDemand = stdDemand

    def genDemand(self):
        demand = max(0, np.random.normal(self.muDemand, self.stdDemand))
        # make sure demand is integer
        demand = np.floor(demand)
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
import tensorflow as tf
from tensorflow.keras import Input, layers, Model, losses, optimizers

class ValueEstimator():
    def __init__(self, α=0.1):
        # self.w = np.zeros(4, )  # state dimension is 3 with 1 bias
        # # self.w = np.zeros(3, )  # state dimension is 3
        # self.α = α
        #
        # self.v_dw = np.zeros(1, )
        # self.β = 0.9
        # self.ε = 1e-10

        inputs = Input(shape=(3,))
        x = inputs
        x = layers.Dense(1000, activation="relu")(x)
        x = layers.Dense(1000, activation="relu")(x)
        x = layers.Dense(1000, activation="relu")(x)
        x = layers.Dense(1000, activation="relu")(x)
        x = layers.Dense(1000, activation="relu")(x)
        x = layers.Dense(1000, activation="relu")(x)
        x = layers.Dense(1000, activation="relu")(x)
        x = layers.Dense(1000, activation="relu")(x)
        x = layers.Dense(1000, activation="relu")(x)
        outputs = layers.Dense(1)(x)
        self.model = Model(inputs=inputs, outputs=outputs)

        # opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0)
        opt = "Adam"
        loss = "mean_squared_error"
        self.model.compile(loss=loss,
                           optimizer=opt)

    def predict(self, s):
        """
        native
        """
        # # add bias
        # s = np.concatenate((s, [1]))
        #
        # value = self.w @ s

        """
        tensorflow-keras
        """
        state = np.reshape(s, (1, 3))
        value = self.model.predict(state)

        return value[0, 0]

    def update(self, s, target):
        """
        tensorflow-keras
        """
        state = np.reshape(s, (1, 3))
        target = np.array([target])

        # oldValue = self.predict(s)
        self.model.fit(state, target, verbose=0)
        # newValue = self.predict(s)
        # print("valueEstimator | s: {}, oldValue: {}, target: {}, newValue: {}".format(s, oldValue, target, newValue))

        """
        native
        """
        # value = self.predict(s)
        #
        # # add bias
        # s = np.concatenate((s, [1]))
        #
        # # update w
        # dloss = 2 * (value - target) * s
        # # # minimize SGD
        # # self.w -= self.α * dloss
        # # # minimize RMSPROP
        # self.v_dw = self.β * self.v_dw + (1 - self.β) * (dloss ** 2)
        # self.w -= self.α * dloss / (np.sqrt(self.v_dw) + self.ε)
        #
        # # print("valueEstimator", s, value, target, dloss, self.w, self.predict(s))
        # print("valueEstimator", s, value, target, self.w @ s)



# class PolicyEstimator():
#     def __init__(self, α=0.01):
#         # assume diagonal covariance
#         # self.θ = np.zeros((3, 3))
#         self.θ = np.zeros((3, 4))
#         self.σ = 10
#         self.α = α
#
#         self.losses = []
#
#         # self.v_dθ = np.zeros((3, 3))
#         self.v_dθ = np.zeros((3, 4))
#         self.β = 0.9
#         self.ε = 1e-10
#
#     def predict(self, s, capacity, retailerOrderQty):
#         # mu = self.mu_state(state)
#
#         """
#         decide in this order:
#         1. reorder point for retailer
#         2. order for node 1
#         3. order for node 0
#         """
#
#         # capacity = state[6]
#         # retailerOrderQty = self.retailerOrder
#         # s = state[0]  # inventory
#
#         s = np.concatenate((s, [1]))
#
#         a = [np.nan, np.nan, np.nan]
#         mu = [np.nan, np.nan, np.nan]
#         # choose reorder point for next cycle
#         # reorder point up to capacity
#         mu[2] = self.θ[2] @ s
#         # mu[2] = self.θ[2] @ np.concatenate((s, [0, 0]))
#         a[2] = np.random.normal(mu[2], self.σ)
#         a[2] = np.clip(a[2], 0, int(retailerOrderQty + s[2]))
#
#         mu[1] = self.θ[1] @ s
#         # mu[1] = self.θ[1] @ np.concatenate((s, [a[2]], [0]))
#         a[1] = np.random.normal(mu[1], self.σ)
#         a[1] = np.clip(a[1], retailerOrderQty-s[1], capacity[1]-s[1])
#         a[1] = np.clip(a[1], 0, None)  # limit to 0
#
#         mu[0] = self.θ[0] @ s
#         # mu[0] = self.θ[0] @ np.concatenate((s, [a[2]], [a[1]]))
#         a[0] = np.random.normal(mu[0], self.σ)
#         a[0] = np.clip(a[0], a[1]-s[0], capacity[0]-s[0])
#         a[0] = np.clip(a[0], 0, None)  # limit to 0
#
#         # discretize actions
#         a = np.floor(a)
#
#         return mu, a
#
#     def update(self, s, target, a, capacity, retailerOrderQty):
#
#         # decide parameter
#         mu, _ = self.predict(s, capacity, retailerOrderQty)
#
#         # loss
#         loss = -np.sum((a - mu)**2) * target
#
#         # update θ
#         # s = np.concatenate((s, a[0:2]))
#         s = np.concatenate((s, [1]))
#         dloss = -np.outer(a - mu, s) * target
#         # # minimize SGD
#         # self.θ -= self.α * dloss
#
#         # minimize RMSPROP
#         self.v_dθ = self.β * self.v_dθ + (1 - self.β) * (dloss ** 2)
#         # element-wise division by sqrt
#         self.θ += self.α * dloss / (np.sqrt(self.v_dθ) + self.ε)
#
#         # newmu = self.predict(s, capacity, retailerOrderQty)[0]
#         # newloss = -np.sum((a - newmu)**2) * target
#         # print("policyEstimator", s, mu, a, target, newmu, loss, newloss)
#
#         # compare loss between old and new θ

"""
with tensorflow-keras
"""
import tensorflow as tf
from tensorflow.keras import Input, layers, Model, losses, optimizers
import tensorflow_probability as tfp


class PolicyEstimator():
    def __init__(self, α=0.01):
        self.stdVal = 5  # default standard deviation for exploration

        inputs = Input(shape=(4,))
        capacity = Input(shape=(3,))
        retailerOrderQty = Input(shape=(1,))
        prevAction = Input(shape=(3,))
        target = Input(shape=(1,))

        x = inputs
        x = layers.Dense(100, activation="relu")(x)
        x = layers.Dense(80, activation="relu")(x)
        x = layers.Dense(60, activation="relu")(x)
        x = layers.Dense(40, activation="relu")(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dense(40, activation="relu")(x)
        x = layers.Dense(60, activation="relu")(x)
        x = layers.Dense(80, activation="relu")(x)
        x = layers.Dense(100, activation="relu")(x)
        mu = layers.Dense(3)(x)
        std = Input(shape=(3,))  # std = 5.0

        normal_dist = tfp.distributions.Normal(mu, std)

        # normal distribution
        # sampling
        def sampling(args):
            return normal_dist.sample(1)

        aRaw = layers.Lambda(sampling, output_shape=(3,))([mu, std])
        # clipping
        s0, s1, s2, _ = tf.split(inputs, 4, axis=-1)
        a0, a1, a2 = tf.split(aRaw, 3, axis=-1)
        c0, c1, c2 = tf.split(capacity, 3, axis=-1)

        # TODO: don't clip the reorder point
        a2 = tf.clip_by_value(a2, 0, 10) #5)  # retailerOrderQty + s2)
        a2 = tf.math.floor(a2)
        a2 = tf.clip_by_value(a2, 0, c2 - s2)  # limit capacity
        a1 = tf.clip_by_value(a1, retailerOrderQty - s1, c1 - s1)
        a1 = tf.math.floor(a1)
        a0 = tf.clip_by_value(a0, a1 - s0, c0 - s0)
        a0 = tf.math.floor(a0)
        a = tf.concat([a0, a1, a2], axis=-1)
        #         a = tf.clip_by_value(a, 0, 30)

        # consider training in a batch for 1-episode, speed up
        self.model = Model(inputs=[inputs, std, capacity, retailerOrderQty, prevAction, target], outputs=[mu, a])

        loss = -normal_dist.log_prob(prevAction) * target
        optimizer = "adam"
        self.model.add_loss(loss)
        self.model.compile(optimizer=optimizer, experimental_run_tf_function=False)

    def predict(self, s, capacity, retailerOrderQty):
        s = np.concatenate((s, [1]))  # add bias term
        s = np.reshape(s, (1, 4))
        _action = np.array([[0, 0, 0]])
        _target = np.array([0])
        capacity = np.array([capacity])
        retailerOrderQty = np.array([retailerOrderQty])
        std = np.array([[self.stdVal, self.stdVal, self.stdVal]])
        mu, a = self.model.predict([s, std, capacity, retailerOrderQty, _action, _target])

        return mu[0], a[0, 0]

    def update(self, s, target, a, capacity, retailerOrderQty):
        s = np.concatenate((s, [1]))  # add bias term
        s = np.reshape(s, (1, 4))
        capacity = np.array([capacity])
        retailerOrderQty = np.array([retailerOrderQty])
        a = np.reshape(a, (1, 3))
        target = np.array([target])
        std = np.array([[self.stdVal, self.stdVal, self.stdVal]])

        # exponent
        target = 10 * np.exp(target / 10000)
        # rescale
        # target /= 100
        # clip gradient
        target = np.clip(target, 0, 30)

        # muBefore = self.predict(s[0, :3], capacity[0], retailerOrderQty[0])[0]
        self.model.fit([s, std, capacity, retailerOrderQty, a, target], epochs=1, verbose=0)
        # muAfter = self.predict(s[0, :3], capacity[0], retailerOrderQty[0])[0]
        # print("s: {}, mu: {}, a: {}, exp_td_error: {}, newMu: {}".format(s[0, :3], muBefore, a, target, muAfter))
        # print("")


class PlannerWithPolicyGradient():
    def __init__(self, learningParams, retailerOrder):
        # create policy and actor
        self.policy_estimator = PolicyEstimator(α=1)
        self.value_estimator = ValueEstimator(α=1)

        # learning params
        # self.alpha = learningParams["alpha"]
        self.epsilon = 1 #learningParams["epsilon"]
        # self.gamma = learningParams["gamma"]
        self.discount_factor = 0.2 #0 #0.95 #0.2

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

        # mu is inaccurate as it's not clipped yet
        _, a = self.policy_estimator.predict(s, capacity, self.retailerOrder)
        # epsilon-greedy
        # if np.random.uniform() < self.epsilon:
        #     # take sampling
        #     _, a = self.policy_estimator.predict(s, capacity, self.retailerOrder)
        # else:
        #     # greedy action (take mu)
        #     a, _ = self.policy_estimator.predict(s, capacity, self.retailerOrder)

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

            # make all rewards positive and change task to maximize reward
            # if reward positive and action differ, then minimise the difference between action and mu
            # make mu closer to that action
            # td_error = 10 * np.exp(td_error / 1000)

            # # update the value estimator
            self.value_estimator.update(old_s, td_target)
            # # update the policy estimator
            capacity = oldState[6]
            self.policy_estimator.update(old_s, td_error, old_a, capacity, self.retailerOrder)

            # print(old_s, value_now, reward, td_target, self.value_estimator.predict(old_s))

        return

import numpy as np
import environment
import agent

def runEpisode(envParams, plannerAgent, train=True):
    # create the environment
    env = environment.Environment(envParams)

    # create demand agent
    demandAgent = agent.DemandAgent(envParams["muDemand"], envParams["stdDemand"])

    # initial state
    state = np.array([
        [0, 0, 10],  # inventory
        [0, 0, 5],  # reorder point
        [0, 0, 0],  # pendingDelivery
        [0, 0, 0],  # daysToDelivery
        [0, 0, 0],  # stockouts
        [1, 3, 0],  # processingTimes
        [30, 30, 30]  # maximum inventory capacity
    ])

    # history tracker
    stateHistory = [state]
    actionHistory = [plannerAgent.resetAction()]
    rewardHistory = [np.nan]
    experiences = []

    # keep track of old state and action
    oldState = None
    oldAction = None

    # play 1 episode of the game for N length
    for i in range(envParams["N"]):

        """
        normal state transition
        """
        # generate demand
        demand = demandAgent.genDemand()

        # assert that inventory is non zero
        # try:
        #     assert state.min() >= 0
        # except:
        #     print(stateHistory, "#")
        #     print("")
        #     print(actionHistory, "#")
        #     print("")
        #     _s = qlearning.array2key(stateHistory[-2, 0, :2])
        #     print(_s, plannerAgent.q.getActions(_s))
        #     raise Exception("invalid move")


        # state transition
        # actionTrigger if retailer hits reorder point
        state1, actionTrigger, reward = env.step(state, demand)

        """
        execute action
        """
        # check if action can be taken
        if actionTrigger:
            # planner decides action
            action = plannerAgent.takeAction(state1, demand)

            # try:
            #     assert state1[0, 0] + action[1, 0] >= action[1, 1]
            #     assert state1[0, 1] + action[1, 1] >= action[1, 2]
            # except:
            #     print(state1, "#")
            #     print("")
            #     print(action, "#")
            #     print("")
            #     _s = qlearning.array2key(state1[0, :2])
            #     print(_s, plannerAgent.q.getActions(_s))
            #     raise Exception("invalid move")

            # execute action
            state2 = env.execute(state1, action)

            # if train
            if train:
                plannerAgent.train(oldState, oldAction, state1, reward)

            experiences.append([oldState, oldAction, state1, reward])

            # keep track of current state and action
            oldState = state1
            oldAction = action

            # set newState
            state = state2

            # keep track of reward
            rewardHistory.append(reward)

        else:
            # reset action
            action = plannerAgent.resetAction()

            # set newState
            state = state1

            # keep track of reward
            rewardHistory.append(np.nan)

        # keep track of state and action for visualization
        stateHistory = np.append(stateHistory, [state], axis=0)
        actionHistory = np.append(actionHistory, [action], axis=0)

    return (stateHistory, actionHistory, rewardHistory, experiences)

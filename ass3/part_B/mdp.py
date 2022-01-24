"""
COL333 Assignment 3 - Modelling a Taxi Domain sequential decision making problem.
We create a simulator for the Taxi Domain environment, modelling it as an MDP.
We then solve the MDP in an offline setting through dynamic programming, and
in an online setting through reinforcement learning.
Written By:
Ramneet Singh (2019CS50445)
Dhairya Gupta (2019CS50---)
"""

RANDOM_SEED = 333445
from numpy.random import default_rng
rng = default_rng(RANDOM_SEED)

import random
random.seed(RANDOM_SEED)

from collections import defaultdict
import math
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

################### GENERAL MDP TOOLS ######################

class MDP(object):
    """
    Base Class for an MDP. Stores the following:
    states - A list of states of the MDP.
    actions - A list of actions.
    initState - Initial state.
    terminalFn - terminalFn(s) returns True iff s is a terminal state
    rewardFn - rewardFn(s,a,s') must return the reward obtained on taking
    action a in state s and reaching state s'
    transitionFn - transitionFn(s,a) returns a list of tuples of the form
    (P(s' | s,a), s') with successor states and their corresponding transition
    probabilities.
    Implements the following:
    takeAction(s,a) - Returns the state reached and the reward obtained in a
    simulation of taking action a in state s.
    """

    def __init__(self, states, actions, initState, terminalFn, rewardFn, transitionFn):
        self.states = states
        self.actions = actions
        self.initState = initState
        self.terminalFn = terminalFn
        self.rewardFn = rewardFn
        self.transitionFn = transitionFn

    def takeAction(self, s, a):
        """Simulate taking an action a in state s. Return (state, reward) obtained."""
        if self.terminalFn(s):
            raise Exception("takeAction called on terminal state!")
        successors = self.transitionFn(s,a)
        # print("[state]")
        # print(s)
        # print("[successors]")
        # print(len(successors))
        successorProbs, successorStates = list(map(lambda v:v[0], successors)), list(map(lambda v:v[1], successors))

        # Toss a coin to find the successor
        result = rng.multinomial(1, successorProbs)
        sPrime = successorStates[list(result).index(1)]
        reward = self.rewardFn(s, a, sPrime)
        return sPrime, reward

################### TAXI DOMAIN TOOLS ######################

class TaxiState(object):

    def __init__(self, taxiPos, pasPos, inTaxi):
        self.taxiPos = taxiPos
        self.pasPos = pasPos
        self.inTaxi = inTaxi

    def __str__(self):
        return (f"[Taxi Position] {self.taxiPos}\n[Passenger Position] {self.pasPos}\n[In Taxi] {self.inTaxi}")

    def __eq__(self, other):
        return (self.taxiPos==other.taxiPos and self.pasPos==other.pasPos and self.inTaxi==other.inTaxi)

    def __hash__(self):
        return hash(str(self))

    def getInTaxi(self):
        return self.inTaxi

    def getPasPos(self):
        return self.pasPos

    def getTaxiPos(self):
        return self.taxiPos

def createTaxiProblem(pasStart, destDepot, taxiStart, grid):
    """
    Create and return a Taxi MDP Instance. Arguments:
    pasStart - Starting position of passenger
    destDepot - Position of destination depot
    taxiStart - Starting position of taxi
    grid - Grid world environment
    """

    # States
    states = []
    positions = grid.getAllPositions()
    for taxiPos in positions:
        for pasPos in positions:
            inState = TaxiState(
                taxiPos = taxiPos,
                pasPos = pasPos,
                inTaxi = True
            )

            outState = TaxiState(
                taxiPos = taxiPos,
                pasPos = pasPos,
                inTaxi = False
            )

            states = states + [inState, outState]

    # Actions
    actions = [
        "N",
        "S",
        "E",
        "W",
        "PICKUP",
        "PUTDOWN"
    ]

    # Initial state
    initState = TaxiState(
        taxiPos = taxiStart,
        pasPos = pasStart,
        inTaxi = False
    )

    # Terminal function
    def terminalFn(s):
        if (s.getTaxiPos() == destDepot) and (s.getPasPos() == destDepot) and (not s.getInTaxi()):
            return True
        return False

    # Reward function
    def rewardFn(s, a, sPrime):
        pasPos, taxiPos, inTaxi = s.getPasPos(), s.getTaxiPos(), s.getInTaxi()

        if a=="PUTDOWN" and (taxiPos==destDepot) and (pasPos==destDepot) and inTaxi:
            return 20
        elif (a=="PUTDOWN" or a=="PICKUP") and (pasPos != taxiPos):
            return -10
        else:
            return -1

    # Transition function
    def transitionFn(s, a):
        pasPos, taxiPos, inTaxi = s.getPasPos(), s.getTaxiPos(), s.getInTaxi()

        if a=="PICKUP":
            if pasPos==taxiPos:
                newState = TaxiState(
                    pasPos = pasPos,
                    taxiPos = taxiPos,
                    inTaxi = True
                )
                return [(1.0, newState)]
            else:
                return [(1.0, s)]
        elif a=="PUTDOWN":
            if pasPos==taxiPos:
                newState = TaxiState(
                    pasPos = pasPos,
                    taxiPos = taxiPos,
                    inTaxi = False
                )
                return [(1.0, newState)]
            else:
                return [(1.0, s)]
        else:
            # Navigation action.
            # Moves with 0.85 probability in the intended direction and with 0.15 in a random other direction.
            newPosProbs = defaultdict(lambda: 0.0)

            for act in ['N', 'S', 'E', 'W']:
                newTaxiPos = grid.move(taxiPos, act)
                if act==a:
                    newPosProbs[newTaxiPos] = newPosProbs[newTaxiPos] + 0.85
                else:
                    newPosProbs[newTaxiPos] = newPosProbs[newTaxiPos] + 0.05

            successors = []
            for pos in newPosProbs:
                if inTaxi:
                    newPasPos = pos
                else:
                    newPasPos = pasPos
                newState = TaxiState(
                    pasPos = newPasPos,
                    taxiPos = pos,
                    inTaxi = inTaxi
                )
                successors.append((newPosProbs[pos], newState))
            return successors
    return MDP(
        states,
        actions,
        initState,
        terminalFn,
        rewardFn,
        transitionFn
    )

################### GRID WORLD TOOLS ########################

class SquareGrid(object):

    def __init__(self, size, walls):
        w = defaultdict(list)

        # Make the corner walls.  -> = x direction = 1st Coordinate of Tuple. Up arrow = y direction = 2nd Coordinate
        w[(0,0)] = ['S', 'W']
        w[(size-1,0)] = ['S', 'E']
        w[(0,size-1)] = ['N', 'W']
        w[(size-1,size-1)] = ['N', 'E']

        # Make the boundary walls
        for i in range(1,size-1):
            w[(0,i)] = ['W']
            w[(i,0)] = ['S']
            w[(size-1,i)] = ['E']
            w[(i, size-1)] = ['N']

        # Make custom walls
        for pos in walls:
            w[pos] = w[pos] + walls[pos]

        self.size = size
        self.walls = w

    def __str__(self):
        """
        Print the grid. For example, the 5x5 grid from the assignment gives the following output:
        + --- --- --- --- --- +
        |  O   O | O   O   O  |
        +                     +
        |  O   O | O   O   O  |
        +                     +
        |  O   O   O   O   O  |
        +                     +
        |  O | O   O | O   O  |
        +                     +
        |  O | O   O | O   O  |
        + --- --- --- --- --- +
        """
        # Print top boundary
        out = "+ "
        out = out + self.size*"--- " + "+" + "\n"

        # Print rows
        for i in range(self.size):
            for j in range(self.size):
                walls = self.walls[(j, self.size-i-1)]
                if 'W' in walls:
                    if j==0:
                        out += "| "
                    else:
                        out += "|"
                if 'E' in walls and j!=(self.size-1):
                    out += " O "
                else:
                    out += " O  "
            out += "|\n"

            out += "+ "
            for j in range(self.size):
                walls = self.walls[(j, self.size-i-1)]
                if 'S' in walls:
                    out += "--- "
                else:
                    out += "    "
            out += "+\n"

        return out

    def getAllPositions(self):
        positions = []
        for i in range(self.size):
            for j in range(self.size):
                positions.append((i,j))
        return positions

    def getAllDepots(self):
        """Return the positions of all depots in the grid."""
        if self.size==5:
            return [(0,0), (0,4), (4,0), (4,4)]
        elif self.size==10:
            return [(0,1), (0,9), (3,6), (4,0), (5,9), (6,5), (8,9), (9,0)]

    def move(self, curPos, direction):
        """From curPos, move in direction. Return the new position."""
        if curPos in self.walls:
            walls = self.walls[curPos]
            if direction in walls:
                return curPos
        curX, curY = curPos[0], curPos[1]

        if direction=='N':
            return (curX, curY+1)
        elif direction=='S':
            return (curX, curY-1)
        elif direction=='E':
            return (curX+1, curY)
        elif direction=='W':
            return (curX-1, curY)

################### LEARNING ALGORITHMS ######################

def QLearningUpdate(QVals, actions, s, a, sPrime, r, alpha, gamma):
    return ( (1-alpha)*QVals[(s, a)] + alpha*( r + gamma*max([QVals[(sPrime,aPrime)] for aPrime in actions]) ) )

def SARSAUpdate(QVals, s, a, r, sPrime, aPrime, alpha, gamma):
    return ( (1-alpha)*QVals[(s,a)] + alpha*( r + gamma*QVals[(sPrime, aPrime)] ) )

def evaluatePolicy(
    reset,
    destination,
    grid,
    policy,
    maxLength=500,
    gamma=0.99,
    numInstances=10):
    """
    Evaluate the policy by taking numInstances random instances and then computing the average sum of discounted rewards.
    maxLength is the max number of steps to wait before we forcefully say an episode is over.
    """
    sumRewards = 0

    for _ in range(numInstances):
        instance = reset(destination, grid)
        state = instance.initState
        terminalFn = instance.terminalFn
        numSteps = 0
        rewards = 0
        discount = 1
        while(True):
            if terminalFn(state) or numSteps==maxLength:
                break
            if state not in policy:
                print("[No State]")
                print(state)
            state, reward = instance.takeAction(state, policy[state])
            rewards += (discount*reward)
            discount *= gamma
            numSteps += 1
        sumRewards += rewards
    return (sumRewards / numInstances)


def learn(
    destination,
    reset,
    algo,
    grid,
    baseEpsilon=0.1,
    strategy='fixed',
    alpha=0.25,
    gamma=0.99,
    episodes=2000,
    maxLength=500,
    avgInstances=10,
    ):
    """
    Generic wrapper function for a model-free sample-based learning algorithm that follows an epsilon greedy strategy.
    Parameters:
    destination - Destination of the passenger.
    reset - Function which gives a fresh episode to run on. reset(destination, grid) returns a new MDP.
    algo - The learning algorithm to use. Currently supports "Q-Learning" and "SARSA"
    baseEpsilon - Epsilon parameter for the epsilon greedy strategy. Trades off exploration vs exploitation
    strategy - Specifies how epsilon is varied with number of iterations of learning.
               Currently supports "fixed" and "decaying".
    alpha - The learning rate. This is the weight given to the new sample estimate versus the old one.
    gamma - Discount factor for future rewards.
    episodes - Number of episodes to use for training.
    maxLength - Max number of steps an episode can go on for before we end it.
    avgInstances - Number of instances to average the accumulated discounted reward over for plotting.
    grid - Grid world environment
    """
    outputFile = "rewardVsEpisodes" + f"_{algo}_" + f"_epsilon{baseEpsilon}_{strategy}_" + f"_grid{grid.size}_" + f"_alpha{alpha}_" + f"_gamma{gamma}_" + ".csv"
    with open(outputFile, 'w') as f:
        QVals = defaultdict(lambda: 0.0) # Initialise all Q Value estimates to 0
        numEpisodes, numUpdates = 0, 0 # Number of episodes, updates
        pbar = tqdm(total=episodes)
        while(True):
            instance = reset(destination, grid)

            curPolicy = getPolicy(instance.states, instance.actions, QVals)
            reward = evaluatePolicy(reset, destination, grid, curPolicy, maxLength, gamma, avgInstances)

            f.write(f"{numEpisodes},{reward}\n")

            if numEpisodes >= episodes:
                return curPolicy, reward

            state = instance.initState
            actions = instance.actions
            terminalFn = instance.terminalFn
            numSteps = 0
            prevState, prevAction, prevReward = None, None, None
            while(True):
                if strategy=="fixed":
                    epsilon = baseEpsilon
                else:
                    epsilon = (baseEpsilon / (numUpdates+1))
                if numSteps > maxLength or terminalFn(state):
                    break
                p = random.random()
                if p < epsilon:
                    action = random.choice(actions)
                else:
                    action = curPolicy[state]

                newState, reward = instance.takeAction(state, action)

                if algo=="Q-Learning":
                    QVals[(state,action)] = QLearningUpdate(QVals, actions, state, action, newState, reward, alpha, gamma)
                elif algo=="SARSA":
                    if prevState is not None:
                        QVals[(prevState, prevAction)] = SARSAUpdate(QVals, prevState, prevAction, prevReward, state, action, alpha, gamma)

                prevState, prevAction, prevReward = state, action, reward
                state = newState
                numSteps += 1
                numUpdates += 1

            numEpisodes += 1
            pbar.update(1)
        pbar.close()


def getPolicy(states, actions, QVals):
    """
    Return the optimal policy (as a {state:action} dict), assuming the QVals estimates are correct.
    Note: This will also assign an action to a terminal state. It is the job of the agent to recognise terminal states.
    """
    policy = {}
    for s in states:
        maxQVal = -math.inf
        maxActions = []
        for a in actions:
            newQVal = QVals.get((s,a), 0)
            if newQVal > maxQVal:
                maxQVal = newQVal
                maxActions = [a]
            elif newQVal == maxQVal:
                maxActions.append(a)
        policy[s] = random.choice(maxActions)
    return policy

def generateRandomEpisode(destination, grid):
    """
    Generate and return a random instance of the Taxi MDP with given destination.
    - Select a passenger start depot randomly from all depots except the destination depot.
    - Select a taxi start state randomly from anywhere in the grid.
    """
    gridSize = grid.size
    pasStart = random.choice([p for p in grid.getAllDepots() if (not (p == destination))])
    taxiStart = random.choice([(i,j) for j in range(gridSize) for i in range(gridSize)])
    taxiMDP = createTaxiProblem(
        pasStart,
        destination,
        taxiStart,
        grid
    )
    return taxiMDP

def plotRewards(
    algo,
    baseEpsilon,
    strategy,
    grid,
    alpha,
    gamma
):
    """Create plot of rewards vs number of episodes."""
    outputFile = "rewardVsEpisodes" + f"_{algo}_" + f"_epsilon{baseEpsilon}_{strategy}_" + f"_grid{grid.size}_" + f"_alpha{alpha}_" + f"_gamma{gamma}_" + ".csv"
    rewardsDf = pd.read_csv(outputFile, header=None)
    plt.plot(rewardsDf.iloc[:,0], rewardsDf.iloc[:,1], 'r-')
    plt.xlabel("Number of Training Episodes")
    plt.ylabel("Average Accumulated Discounted Reward")
    plt.title(f"Reward vs Number of Training Episodes ({algo})\nEpsilon={baseEpsilon} {strategy}, gridSize={grid.size}\nalpha={alpha}, gamma={gamma}")

    plotFile = "rewardVsEpisodes" + f"_{algo}_" + f"_epsilon{baseEpsilon}_{strategy}_" + f"_grid{grid.size}_" + f"_alpha{alpha}_" + f"_gamma{gamma}_" + ".png"
    plt.savefig(plotFile, dpi=300)

################### DRIVER CODE ##############################

walls5x5 = {
    (0,0): ['E'],
    (1,0): ['W'],
    (2,0): ['E'],
    (3,0): ['W'],
    (0,1): ['E'],
    (1,1): ['W'],
    (2,1): ['E'],
    (3,1): ['W'],
    (1,3): ['E'],
    (2,3): ['W'],
    (1,4): ['E'],
    (2,4): ['W']
}

walls10x10 = {
    (0,0): ['E'],
    (0,1): ['E'],
    (0,2): ['E'],
    (0,3): ['E'],
    (1,0): ['W'],
    (1,1): ['W'],
    (1,2): ['W'],
    (1,3): ['W'],
    (3,0): ['E'],
    (3,1): ['E'],
    (3,2): ['E'],
    (3,3): ['E'],
    (4,0): ['W'],
    (4,1): ['W'],
    (4,2): ['W'],
    (4,3): ['W'],
    (7,0): ['E'],
    (7,1): ['E'],
    (7,2): ['E'],
    (7,3): ['E'],
    (8,0): ['W'],
    (8,1): ['W'],
    (8,2): ['W'],
    (8,3): ['W'],
    (2,6): ['E'],
    (2,7): ['E'],
    (2,8): ['E'],
    (2,9): ['E'],
    (3,6): ['W'],
    (3,7): ['W'],
    (3,8): ['W'],
    (3,9): ['W'],
    (7,6): ['E'],
    (7,7): ['E'],
    (7,8): ['E'],
    (7,9): ['E'],
    (8,6): ['W'],
    (8,7): ['W'],
    (8,8): ['W'],
    (8,9): ['W'],
    (5,4): ['E'],
    (5,5): ['E'],
    (5,6): ['E'],
    (5,7): ['E'],
    (6,4): ['W'],
    (6,5): ['W'],
    (6,6): ['W'],
    (6,7): ['W']
}

grid5x5 = SquareGrid(size=5, walls=walls5x5)
grid10x10 = SquareGrid(size=10, walls=walls10x10)

if __name__=="__main__":

    # Parts B 2,4
    # policy, _ = learn(
    #     (4,4),
    #     generateRandomEpisode,
    #     "SARSA",
    #     grid = grid5x5,
    #     baseEpsilon=0.1,
    #     strategy='decaying',
    #     alpha=0.25,
    #     gamma=0.99,
    #     episodes=2000,
    #     maxLength=500,
    #     avgInstances=100)
    # plotRewards(
    #     "SARSA",
    #     0.1,
    #     "decaying",
    #     grid5x5,
    #     0.25,
    #     0.99
    # )

    # Part B 3
    # policy, _ = learn(
    #     (4,4),
    #     generateRandomEpisode,
    #     "SARSA",
    #     grid = grid5x5,
    #     baseEpsilon=0.1,
    #     strategy='fixed',
    #     alpha=0.25,
    #     gamma=0.99,
    #     episodes=2000,
    #     maxLength=500,
    #     avgInstances=100)
    # reward = evaluatePolicy(
    #     generateRandomEpisode,
    #     dest,
    #     grid=grid5x5,
    #     policy=policy,
    #     maxLength=500,
    #     gamma=0.99,
    #     numInstances=5
    # )
    # print(f"Average Reward over 5 Random Instances is {reward:.3f}")


    # Part B 5
    destinations = [(0,1), (3,6), (6,5), (8,9), (4,0)]

    sumRewards = 0
    for dest in destinations:
        # TODO Select best learning agent to use here
        policy, _ = learn(
            dest,
            generateRandomEpisode,
            "Q-Learning",
            grid=grid10x10,
            baseEpsilon=0.1,
            strategy='fixed',
            alpha=0.25,
            gamma=0.99,
            episodes=10000,
            maxLength=500,
            avgInstances=100
        )

        reward = evaluatePolicy(
            generateRandomEpisode,
            dest,
            grid=grid10x10,
            policy=policy,
            maxLength=500,
            gamma=0.99,
            numInstances=500
        )

        print(f"[Agent Learned with Destination {dest}]")
        print(f"[Accumulated Discounted Reward Averaged over 500 instances] = {reward:.3f}")
        sumRewards += reward

    print(f"Average of average rewards of agents trained on 5 destinations = {(sumRewards/5):.3f}")

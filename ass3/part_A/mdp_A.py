"""
COL333 Assignment 3 - Modelling a Taxi Domain sequential decision making problem.
We create a simulator for the Taxi Domain environment, modelling it as an MDP.
We then solve the MDP in an offline setting through dynamic programming, and
in an online setting through reinforcement learning.
Written By:
Ramneet Singh (2019CS50445)
Dhairya Gupta (2019CS50428)
"""
# TODO Complete Dhairya's Entry No.

import sys
import time
import numpy as np
from numpy.random import default_rng
import numpy.random as random
import matplotlib.pyplot as plt
rng = default_rng(33345)

from collections import defaultdict

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
        successors = self.transitionFn(s,a)
        successorProbs, successorStates = list(map(lambda v:v[0], successors)), list(map(lambda v:v[1], successors))

        # Toss a coin to find the successor
        result = list(rng.multinomial(1, successorProbs))
        sPrime = successorStates[result.index(1)]
        reward = self.rewardFn(s, a, sPrime)
        #if(sPrime.getParameters() == s.getParameters()): print(f'successors: {list(map(lambda v:v[0], successors)), list(map(lambda v:v[1].getParameters(), successors))}')
        return sPrime, reward

################### TAXI DOMAIN TOOLS ######################

class TaxiState(object):

    def __init__(self, taxiPos, pasPos, inTaxi):
        self.taxiPos = taxiPos
        self.pasPos = pasPos
        self.inTaxi = inTaxi

    def getInTaxi(self):
        return self.inTaxi

    def getPasPos(self):
        return self.pasPos

    def getTaxiPos(self):
        return self.taxiPos
    def getParameters(self):
        return (self.taxiPos, self.pasPos, self.inTaxi)
    def __str__(self):
        return (f'TaxiPos: {self.taxiPos}, PasPos: {self.pasPos}, InTaxi: {self.inTaxi}')
    

def createTaxiProblem(pasStart, destDepot, taxiStart, grid):
    """
    Create and return a Taxi MDP Instance. Arguments:
    pasStart - Starting position of passenger
    destDepot - Position of destination depot
    taxiStart - Starting position of taxi
    grid - Grid world environment
    """

    # States
    states = {}
    positions = grid.getAllPositions()
    for taxiPos in positions:
        for pasPos in positions:
            inState = None
            if taxiPos == pasPos:
                inState = TaxiState(
                    taxiPos = taxiPos,
                    pasPos = pasPos,
                    inTaxi = True
                )

                states[(taxiPos, pasPos, True)] = inState

            outState = TaxiState(
                taxiPos = taxiPos,
                pasPos = pasPos,
                inTaxi = False
            )

            states[(taxiPos, pasPos, False)] = outState

            '''##states = states + [inState, outState]
            if inState != None: states = states + [inState, outState]
            else: states = states + [outState]'''

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
    '''initState = TaxiState(
        taxiPos = taxiStart,
        pasPos = pasStart,
        inTaxi = False
    )'''
    initState = states[(taxiStart, pasStart, False)]

    # Terminal function
    def terminalFn(s):
        if (s.getTaxiPos() == destDepot) and (s.getPasPos() == destDepot) and (not s.getInTaxi()):
            return True
        return False

    # Reward function
    def rewardFn(s, a, sPrime = None):
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
            if pasPos==taxiPos and (not inTaxi):
                '''newState = TaxiState(
                    pasPos = pasPos,
                    taxiPos = taxiPos,
                    inTaxi = True
                )'''
                newState = states[(taxiPos, pasPos, True)]
                return [(1.0, newState)]
            else:
                return [(1.0, s)]
        elif a=="PUTDOWN":
            if pasPos==taxiPos and inTaxi:
                '''newState = TaxiState(
                    pasPos = pasPos,
                    taxiPos = taxiPos,
                    inTaxi = False
                )'''
                newState = states[(taxiPos, pasPos, False)]

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
                '''newState = TaxiState(
                    pasPos = newPasPos,
                    taxiPos = pos,
                    inTaxi = inTaxi
                )'''
                newState = states[(pos, newPasPos, inTaxi)]
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

    def move(self, curPos, direction):
        """From curPos, move in direction. Return the new position."""
        return_val = None
        if curPos in self.walls:
            
            walls = self.walls[curPos]
            if direction in walls:
                return_val = curPos
            
        curX, curY = curPos[0], curPos[1]

        if not return_val:
            if direction=='N':
                return_val = (curX, curY+1)
            elif direction=='S':
                return_val = (curX, curY-1)
            elif direction=='E':
                return_val = (curX+1, curY)
            elif direction=='W':
                return_val = (curX-1, curY)
        return return_val

def ValueIteration(mdp, eps = 1e-3, gamma = 0.9, max_iter = 10000):
    max_norm_arr = []
    state_value = {k: 0 for k in mdp.states.keys()}
    state_value_prev = state_value.copy()

    for iter in range(max_iter):
        delta = 0
        for k in mdp.states.keys():
            state = mdp.states[k]
            maxval = -np.inf
            max_action = None
            for action in mdp.actions:
                reward = mdp.rewardFn(state, action)
                successors = mdp.transitionFn(state, action)
                sum = 0
                for s in successors:
                    T_s_a_sprime = s[0]
                    R_s_a_sprime = reward
                    new_state = s[1]
                    tp, pp, it = new_state.getTaxiPos(), new_state.getPasPos(), new_state.getInTaxi()
                    sum+=(T_s_a_sprime * (R_s_a_sprime + gamma*state_value_prev[(tp, pp, it)]))
                if sum > maxval:
                    maxval = sum
                    max_action = action
            state_value[k] = maxval

            if delta < np.abs(state_value[k] - state_value_prev[k]):
                delta = np.abs(state_value[k] - state_value_prev[k])
        max_norm_arr.append(delta)
        if (delta < eps*(1 - gamma)/gamma): break
        state_value_prev = state_value.copy()
    return max_norm_arr, state_value


def PolicyIteration(mdp, linalg = False, gamma = 0.1, policy_loss_thresh = 1e-4, max_iter = 10000):
    states_arr = list(mdp.states.keys())
    n_states = len(states_arr)
    s2idx = {s:idx for idx, s in enumerate(states_arr)}
    def PolicyEvaluationIterative(mdp, policy, gamma = gamma, max_iter = 10000, eps = 1e-3):
        state_value = {k: 0 for k in mdp.states.keys()}
        state_value_prev = state_value.copy()
        for iter in range(max_iter):
            delta = 0
            for k in mdp.states.keys():
                state = mdp.states[k]
                state_params = state.getParameters()
                action = policy[k]
                reward = mdp.rewardFn(state, action)
                successors = mdp.transitionFn(state, action)
                sum = 0
                for s in successors:
                    T_s_a_sprime = s[0]
                    R_s_a_sprime = reward
                    new_state = s[1]
                    tp, pp, it = new_state.getParameters()
                    sum+=(T_s_a_sprime * (R_s_a_sprime + gamma*state_value_prev[(tp, pp, it)]))
                state_value[state_params] = sum
                val1 = np.abs(state_value[state_params] - state_value_prev[state_params])
                if val1 > delta:
                    delta = val1
                
                
            assert list(state_value.keys()) == list(state_value_prev.keys())
            a1 = np.array(list(state_value.values()))
            a2 = np.array(list(state_value_prev.values()))
            state_value_prev = state_value.copy()

            if(delta < eps * (1 - gamma)/gamma): break
        return state_value, np.array(list(state_value.values())).reshape((-1,1))
    
    def PolicyEvaluationLinAlg(mdp, policy, gamma = gamma):
        R = np.array([mdp.rewardFn(mdp.states[k], policy[k]) for k in mdp.states.keys()]).reshape((-1, 1))
        T_mat = np.zeros((n_states, n_states))
        for k in mdp.states:
            state = mdp.states[k]
            action = policy[k]
            successors = mdp.transitionFn(state, action)
            for s in successors:
                tp, pp, it = s[1].getParameters()
                T_mat[s2idx[k], s2idx[(tp, pp, it)]] = s[0]
        #R = R_vec.reshape((-1, 1))
        rhs = -(R*T_mat)@np.ones((n_states, 1))
        lhs_mat = (gamma*T_mat - np.identity(n_states))
        util_vec = (np.linalg.inv(lhs_mat)@rhs).reshape((-1, ))
        value_dict = {s: util_vec[idx] for idx, s in enumerate(states_arr)}
        return value_dict, util_vec

    def PolicyImprovement(mdp, value_dict, prev_policy, gamma = gamma):
        policy_dict = {k:None for k in mdp.states.keys()}
        changed = False
        for k in mdp.states.keys():
            state = mdp.states[k]
            maxval = -np.inf
            max_action = None
            for action in mdp.actions:
                successors = mdp.transitionFn(state, action)
                reward = mdp.rewardFn(state, action)
                sum = 0
                for s in successors:
                    tp, pp, it = s[1].getTaxiPos(), s[1].getPasPos(), s[1].getInTaxi()
                    sum+=(s[0]*(reward + gamma*value_dict[(tp, pp, it)]))
                if(sum>maxval):
                    maxval = sum
                    max_action = action
            policy_dict[k] = max_action
            if(prev_policy[k] != max_action): changed = True
        return policy_dict, changed

    def randomPolicyInit():
        n = len(mdp.actions)
        policy = {k: mdp.actions[random.randint(0, n)] for k in mdp.states}
        return policy

    
    policy = randomPolicyInit()
    #prev_policy = policy
    prev_value_dict = {k: 0 for k in mdp.states}
    prev_value_np = np.array(list(prev_value_dict.values())).reshape((-1, ))

    changed = True

    it = 0
    max_it = max_iter
    policy_loss_arr = []
    policy_loss_thresh = policy_loss_thresh

    t1 = time.time()
    while(changed and it<max_it):
        it+=1
        if(linalg):
            value_dict, value_np = PolicyEvaluationLinAlg(mdp, policy)
        else:
            value_dict, value_np = PolicyEvaluationIterative(mdp, policy)
        policy_loss = np.max(np.abs(value_np - prev_value_np))
        print(policy_loss)
        prev_value_np = value_np.copy()
        policy_loss_arr.append(policy_loss)

        policy, changed = PolicyImprovement(mdp, value_dict, policy)
        if(policy_loss < policy_loss_thresh): break


    t2 = time.time()
    return policy_loss_arr, policy, t2 - t1


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




grid5x5 = SquareGrid(size=5, walls=walls5x5)
depots = {'R': (0, 4), 'G': (4, 4), 'B': (3, 0), 'Y': (0, 0)}
pasDepot = 'Y'
targetDepot = 'G'
taxiStartDepot = 'R'
mdp = createTaxiProblem(depots[pasDepot], depots[targetDepot], depots[taxiStartDepot], grid5x5)
initState = mdp.initState
print(grid5x5)



        
def main_p1(action_seq = None):
    print(f'Initial State --> {initState}')
    if(action_seq == None): action_seq = ["N", "E", "W", "S", "PICKUP", "PUTDOWN"]
    print()
    for action in action_seq:
        state, reward = mdp.takeAction(state, action)
        print(f'Action: {action}')
        print(f'nextState --> {state}')
        print(f'reward: {reward}')
        print()

def main_p2_a(gamma = 0.9, eps = 1e-3):
    print(f'Initial State --> {initState}')
    gamma = 0.9
    eps = 1e-3
    max_norm_arr, _ = ValueIteration(mdp, gamma = gamma, eps = eps)
    print(f'No. of iterations: {len(max_norm_arr)}')
    fig, ax = plt.subplots()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('max-norm distance')
    ax.plot(max_norm_arr)
    ax.set_title(f'Value Iteration; gamma = {gamma}, epsilon = {eps}, iterations = {len(max_norm_arr)}')
    fig.savefig('mdp_Q2a.png')
    return fig, ax

def main_p2_b():
    print(f'Initial State --> {initState}')
    gamma_arr = [0.01, 0.1, 0.5, 0.8, 0.99]
    max_norm_arr_list = [None]*len(gamma_arr)
    eps = 1e-3
    for i, gamma in enumerate(gamma_arr):
        max_norm_arr_list[i], _ = ValueIteration(mdp, gamma = gamma, eps = eps)
        print(f'For gamma = {gamma}, Iterations = {len(max_norm_arr_list[i])}')
    figs = []
    for i in range(len(gamma_arr)):
        fig, ax = plt.subplots()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('max-norm distance')
        ax.set_title(f'Value Iteration; gamma = {gamma_arr[i]}')
        ax.plot(max_norm_arr_list[i])
        fig.savefig(f'mdp_Q2b_gamma_{gamma_arr[i]}.png')
        figs.append(fig)
    return figs

def main_p2_c():
    targetDepot = 'G'
    otherDepots = ['R', 'B', 'Y']
    gamma_arr = [0.1, 0.99]
    n_steps = 20

    for gamma in gamma_arr:
        for td in otherDepots:
            for pd in otherDepots:
                if(pd == td): continue
                print(f'TaxiStartDepot: {td, depots[td]}; PassengerStartDepot: {pd, depots[pd]}; DestinationDepot: {targetDepot, depots[targetDepot]}; gamma = {gamma}')
                mdp = createTaxiProblem(depots[pd], depots[targetDepot], depots[td], grid5x5)
                _, value_dict = ValueIteration(mdp, gamma = gamma)
                state = mdp.initState
                print()
                for step in range(n_steps+1):
                    print(f'At t = {step}, state --> {state}')
                    if(mdp.terminalFn(state)): print(f'END\n'); break
                    maxval = -np.inf
                    max_action = None
                    for action in mdp.actions:
                        sum = 0
                        successors = mdp.transitionFn(state, action)
                        reward = mdp.rewardFn(state, action)
                        for s in successors:
                            tp, pp, it = s[1].getTaxiPos(), s[1].getPasPos(), s[1].getInTaxi()
                            sum += s[0]*(reward + gamma*value_dict[(tp, pp, it)])
                        if(sum>maxval):
                            maxval = sum
                            max_action = action
                    state, reward1 = mdp.takeAction(state, max_action)
                    print(f'Action Taken: {max_action}')
                    print(f'Reward: {reward1}')
                    print()

def main_p3_a(gamma = 0.99, n_steps = 30, linalg = False):
    gamma = gamma
    n_steps = n_steps
    policy_loss_arr, policy, duration = PolicyIteration(mdp, gamma = gamma, linalg = linalg)
    print(f'Policy iterations: {len(policy_loss_arr)}'); print()
    state = mdp.initState
    print(f'TaxiStartDepot: {taxiStartDepot, depots[taxiStartDepot]}, PassengerStartDepot: {pasDepot, depots[pasDepot]}, DestinationDepot: {targetDepot, depots[targetDepot]}, gamma: {gamma}'); print()
    for step in range(n_steps):
        print(f'At time t = {step}, state --> {state}')
        if(mdp.terminalFn(state)): print('END\n'); break
        action = policy[state.getParameters()]
        state, reward = mdp.takeAction(state, action)
        print(f'Action taken: {action}')
        print(f'Reward: {reward}')

def main_p3_b(linalg = False):
    gamma_arr = [0.01, 0.1, 0.5, 0.8, 0.99]
    policy_loss_arr = [None]*len(gamma_arr)
    figs = []
    for i, gamma in enumerate(gamma_arr):
        policy_loss_arr[i], policy, duration = PolicyIteration(mdp, gamma = gamma, linalg = linalg)
        print(f'For gamma = {gamma}, policy_iterations = {len(policy_loss_arr)}')
        fig, ax = plt.subplots()
        ax.plot(policy_loss_arr[i])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Policy loss')
        ax.set_title(f'Policy Loss vs Iterations; gamma = {gamma}')
        fig.savefig(f'mdp_Q3b_gamma_{gamma}.png')
        figs.append(fig)
    return figs
    




        
        



                











        


                    

                
        












if __name__=="__main__":
    
    
    print()
    '''Part 1'''
    #action_seq = ["N", "E", "W", "S", "PICKUP", "PUTDOWN"]
    #_ = main_p1(action_seq = action_seq)
    
    '''Part 2'''
    #_ = main_p2_a()
    #_ = main_p2_b()
    _ = main_p2_c()

    '''Part 3'''
    #_ = main_p3_a(linalg = True)
    #_ = main_p3_b(linalg = True)
    plt.show()




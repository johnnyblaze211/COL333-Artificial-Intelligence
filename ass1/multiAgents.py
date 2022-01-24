# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
from queue import Queue
import random, util
import sys

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        currentPacmanDirection = currentGameState.getPacmanState().getDirection()

        score = 0.
        if action == 'Stop':
            return -100000000.
        newPos = successorGameState.getPacmanPosition()
        xp, yp = newPos
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newFoodNumCount = successorGameState.getNumFood()

        
        "*** YOUR CODE HERE ***"
        newGhostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        
        score += currentGameState.getScore()
        #score += 10000.

        farGhosts = True
        score1 = 0 
        for idx, gpos in enumerate(newGhostPositions):
            xg, yg = gpos   
            if(not newScaredTimes[idx] > 0):
                if (abs(xp - xg) + abs(yp - yg)) < 4:
                    farGhosts = False
                if (abs(xp - xg) + abs(yp - yg)) < 4:
                    score1 += -10000000./2**(abs(xp - xg) + abs(yp - yg) + .001)
                #else:
                    #score1 += -1000./((abs(xp - xg) + abs(yp - yg)) + .001)
        score += score1
        val = 0
        closestFood = 10000, 10000
        minDist = 10000000
        for x in range(newFood.width):
            for y in range(newFood.height):
                if newFood[x][y] == True:
                    if(abs(x - xp) + abs(y - yp) < minDist):
                        closestFood = x, y
                        minDist = abs(x - xp) + abs(y - yp)
                    #val += 10000 - (abs(x - xp) + abs(y - yp) + .01)**2
        
        if currentGameState.getNumFood() - successorGameState.getNumFood() > 0:
            val = 100000 - 1000*minDist if minDist!=10000000 else 100000
        else:
            val = -1000*minDist
        #if newFood[xp][yp]:
            #score+=100000
        if farGhosts:
            score += val/newFoodNumCount if newFoodNumCount != 0 else 10000.

        if farGhosts:
            score += 20*random.randint(1, 5)



        


        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def evalMinNode(self, gameState, level, numGhosts, ghostIndex):
        #print(f'evalMinNode({gameState, level, numGhosts, ghostIndex})')
        if gameState.isWin() or gameState.isLose():
            return None, scoreEvaluationFunction(gameState)
        actions = gameState.getLegalActions(agentIndex = ghostIndex)
        minval = 1e9
        nextAction = None
        if ghostIndex == numGhosts:
            for action in actions:
                successorGameState = gameState.generateSuccessor(ghostIndex, action)
                _, nodeval = self.evalMaxNode(successorGameState, level + 1, numGhosts)
                if nodeval < minval:
                    minval = nodeval
                    nextAction = action
        else:
            for action in actions:
                successorGameState = gameState.generateSuccessor(ghostIndex, action)
                _, nodeval = self.evalMinNode(successorGameState, level, numGhosts, ghostIndex + 1)
                if nodeval < minval:
                    minval = nodeval
                    nextAction = action
        return nextAction, minval

    def evalMaxNode(self, gameState, level, numGhosts):
        #print(f'evalMaxNode({gameState, level, numGhosts})')
        if gameState.isWin() or gameState.isLose():
            return None, scoreEvaluationFunction(gameState)
        if level == self.depth:
            return None, scoreEvaluationFunction(gameState)
        actions = gameState.getLegalActions(0)
        maxval = -1e9
        nextAction = None
        for action in actions:
            successorGameState = gameState.generateSuccessor(0, action)
            _, nodeval = self.evalMinNode(successorGameState, level, numGhosts, ghostIndex = 1)
            if nodeval >= maxval:
                maxval = nodeval
                nextAction = action
        return nextAction, maxval
    def getAction(self, gameState):
        
        numGhosts = gameState.getNumAgents() - 1

        action, _ = self.evalMaxNode(gameState, 0, numGhosts)
        return action

        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def evalMaxNode(self, gameState, alpha1, beta1, level, numGhosts):
        alpha = alpha1
        beta = beta1
        if gameState.isWin() or gameState.isLose() or level == self.depth:
            return None, self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(0)
        maxval = -1e9
        nextAction = None
        for action in actions:
            successorGameState = gameState.generateSuccessor(0, action)
            _, nodeval = self.evalMinNode(successorGameState, alpha, beta, level, numGhosts, 1)
            if nodeval > maxval: 
                maxval = nodeval
                nextAction = action
            if beta < nodeval: return action, nodeval
            if nodeval > alpha: alpha = nodeval
        return nextAction, maxval
    
    def evalMinNode(self, gameState, alpha1, beta1, level, numGhosts, ghostIndex):
        alpha = alpha1
        beta = beta1
        if gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)
        
        actions = gameState.getLegalActions(ghostIndex)
        minval = 1e9
        nextAction = None
        if ghostIndex == numGhosts:
            for action in actions:
                successorGameState = gameState.generateSuccessor(ghostIndex, action)
                _, nodeval = self.evalMaxNode(successorGameState, alpha, beta, level + 1, numGhosts)
                if nodeval < minval:
                    minval = nodeval
                    nextAction = action
                if alpha > nodeval: return nextAction, minval
                if nodeval < beta: beta = nodeval
        else:
            for action in actions:
                successorGameState = gameState.generateSuccessor(ghostIndex, action)
                _, nodeval = self.evalMinNode(successorGameState, alpha, beta, level, numGhosts, ghostIndex + 1)
                if nodeval < minval:
                    minval = nodeval
                    nextAction = action
                if alpha > nodeval: return nextAction, minval
                if nodeval < beta: beta = nodeval
        
        return nextAction, minval
        


    def getAction(self, gameState):
        alpha = -1e9
        beta = 1e9
        numGhosts = gameState.getNumAgents() - 1
        #print(f'NumGhosts: {numGhosts}')
        action, _ = self.evalMaxNode(gameState, alpha, beta, 0, numGhosts)
        return action
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def evalMaxNode(self, gameState, level, numGhosts):
        if gameState.isWin() or gameState.isLose() or self.depth == level:
            return None, self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(0)
        maxval = -1e9
        nextAction = None
        for action in actions:
            successorGameState = gameState.generateSuccessor(0, action)
            _, nodeval = self.evalChanceNode(successorGameState, level, numGhosts, 1)
            if nodeval > maxval:
                maxval = nodeval
                nextAction = action
        return nextAction, maxval
    
    def evalChanceNode(self, gameState, level, numGhosts, ghostIndex):
        if gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(ghostIndex)
        sum = 0
        if numGhosts == ghostIndex:
            for action in actions:
                successorGameState = gameState.generateSuccessor(ghostIndex, action)
                _, nodeval = self.evalMaxNode(successorGameState, level + 1, numGhosts)
                sum += nodeval
        else:
            for action in actions:
                successorGameState = gameState.generateSuccessor(ghostIndex, action)
                _, nodeval = self.evalChanceNode(successorGameState, level, numGhosts, ghostIndex + 1)
                sum += nodeval
        avg = sum/len(actions)
        return _, avg


    def getAction(self, gameState):

        numGhosts = gameState.getNumAgents() - 1

        action, _ = self.evalMaxNode(gameState, 0, numGhosts)
        return action

        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    For the evaluation function, we have various kinds of reward scores. 
    P1. If game is won/lost, a huge +ve/-ve reward is given. 
    P2. numFoodScore is given as (-100)*numFood. Thus, the reward for eating a food pellet is +100.0
    P3. A huge reward of +20000 is given for consuming a foodCapsule {scoreCap = -20000*lenCapsules}
    P4. A reward of +4000 is given for eating a scaredGhost. {score+= (-4000)*scaredGhostCount}
    P5. A huge penalty is given if Non-scared ghosts are too near.
    
    There are 3 possible modes:
    Mode1. There is atleast 1 scaredGhost
        In this case, Pacman Agent targets the closest ScaredGhost
    Mode2. There are no scared ghosts and no. of capsules > 0
        In this case, the target for Pacman is the closestCapsule
    Mode3. There are no scared Ghosts and no capsules
        In this case, the target for Pacman Agent is the nearest food particle.

    The targeting is done by calculating the BFS distance to the target from the current position, 
    and giving a lower penalty for smaller distance. The complexity of this is O(w*h), where w and h are board width and height. 
    This was done taking care  of the --timeout=30 parameter in various layouts. Timeout did not occur for as high as 4 ghosts. 
    Since a non-BFS approach would still be O(m*n){due to calculation of food positions},
    this was found to be a desirable high scoring approach without leading to timeout.


    

    """
    "*** YOUR CODE HERE ***"

    score = 0
    init_score = scoreEvaluationFunction(currentGameState)/10
    score+=init_score

    '''Score for win/lose. See P1 in description'''
    if currentGameState.isWin():
        score += 100000000.
        return score
    if currentGameState.isLose():
        score += -100000000.
        return score

    numGhosts = currentGameState.getNumAgents() - 1
    xp, yp = currentGameState.getPacmanPosition()
    numFood = currentGameState.getNumFood()
    foodMatrix = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = [ghostState.getPosition() for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()
    walls = currentGameState.getWalls()
    lenCapsules = len(capsules)

    '''Reward for eating food pellets. See P2 in description'''
    numFoodScore = - numFood*100
    score+=numFoodScore


    

    '''Reward gained for eating capsules. See P3 in description'''
    scoreCap = -20000*lenCapsules
    if numGhosts >=5:
        scoreCap = -(numGhosts+1)*4000*lenCapsules
    score += scoreCap

    ##Calculation for Closest Scared Ghost by Manhattan distance
    minDistScaredGhost = 1000
    xsg, ysg = None, None       ## x,y values for closest Scared Ghost
    normalGhostScore = 0
    scaredGhostCount = 0
    for (xg, yg), t in zip(ghostPositions, scaredTimes):
        if t>0:
            scaredGhostCount+=1
            dist1 = abs(xg - xp) + abs(yg - yp)
            if dist1 < minDistScaredGhost:
                xsg, ysg = xg, yg
                minDistScaredGhost = dist1
        else:
            dist = abs(xp - xg) + abs(yp - yg)
            if dist < 1:
                normalGhostScore += -100000./(dist+0.1)

    ##Calculation for Closest Capsule by Manhattan Distance
    xcap, ycap = None, None     ## x,y values for closest Capsule
    minDistCap = 1000
    if xsg == None:
        if lenCapsules > 0:
            for (xcap1, ycap1) in capsules:
                dist = abs(xcap1 - xp) + abs(ycap1 - yp)
                if dist < minDistCap:
                    (xcap, ycap) = xcap1, ycap1
                    minDistCap = dist


    '''Reward for Eating Scared Ghost. See P4'''
    score+=(-4000)*scaredGhostCount

    '''Penalty for being near Normal Ghost. See P5'''
    score+= normalGhostScore
    
    

    ###Function for bfs distance from source to target point
    def bfs_dist(xp, yp, walls, food, xt = -1, yt = -1):
        w = walls.width
        h = walls.height
        dict = {}
        q = Queue(maxsize = w*h)
        q.put((xp, yp, 0))
        dict[(xp, yp)] = 1

        while not q.empty():
            (x1, y1, d) = q.get()
            
            if xt == -1:
                if food[x1][y1]:
                    return d
            else:
                if x1 == xt and y1 == yt:
                    return d
            if dict.get((x1-1, y1)) == None and not walls[x1 - 1][y1]: 
                q.put((x1-1, y1, d+1))
                dict[(x1-1, y1)] = 1
            if dict.get((x1+1, y1)) == None and not walls[x1 + 1][y1]: 
                q.put((x1+1, y1, d+1))
                dict[(x1+1, y1)] = 1
            if dict.get((x1, y1-1)) == None and not walls[x1][y1 - 1]: 
                q.put((x1, y1-1, d+1))
                dict[(x1, y1-1)] = 1
            if dict.get((x1, y1+1)) == None and not walls[x1][y1 + 1]: 
                q.put((x1, y1+1, d+1))
                dict[(x1, y1+1)] = 1

    

    ##Reward for smaller BFS distance in mode1. See Mode1 in description
    if xsg!=None: 
        score+= (-100)*bfs_dist(xp, yp, walls, foodMatrix, xt = int(xsg), yt = int(ysg))
    ##Reward for smaller BFS distance in mode2. See Mode1 in description
    elif xsg == None and xcap!=None: 
        score+= (-100)*bfs_dist(xp, yp, walls, foodMatrix, xt = int(xcap), yt = int(ycap))
    ##Reward for smaller BFS distance in mode2. See Mode1 in description
    else:
        score+=(-2)*bfs_dist(xp, yp, walls, foodMatrix)
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

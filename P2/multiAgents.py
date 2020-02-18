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
import random, util

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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPos = [ghostState.getPosition() for ghostState in newGhostStates]
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        GhostStates = currentGameState.getGhostStates()
        ghostPos = [ghostState.getPosition() for ghostState in GhostStates]

        nextMinDistance = min(list(map(lambda x: abs(newPos[0] - x[0]) + abs(newPos[1] - x[1]), newGhostPos)))
        if len(newFood.asList()) == 0:
            return 1000000000000
        nextMinFood = min(list(map(lambda x: abs(newPos[0] - x[0]) + abs(newPos[1] - x[1]), newFood.asList())))
        sumFood = sum(list(map(lambda x: abs(newPos[0] - x[0]) + abs(newPos[1] - x[1]), newFood.asList())))

        if sum(newScaredTimes) > 0:
            scaredStates =  list(filter(lambda x: x.scaredTimer > 0, newGhostStates))
            scaredGhosts = [ghostState.getPosition() for ghostState in scaredStates]
            nextMinDistance = min(list(map(lambda x: abs(newPos[0] - x[0]) + abs(newPos[1] - x[1]), scaredGhosts)))
            notScaredStates = list(filter(lambda x: x not in scaredStates, newGhostStates))
            if len(notScaredStates) == 0:
                nextNotScaredMinDistance = 1
            else:
                nextNotScaredMinDistance = min(list(map(lambda x: abs(newPos[0] - x[0]) + abs(newPos[1] - x[1]), [ghost.getPosition() for ghost in notScaredStates])))
            if nextMinDistance == 0:
                return 1000000000
            return (nextNotScaredMinDistance)*abs(successorGameState.getScore())**2 / (sumFood * len(newFood.asList()) * (nextMinDistance+1))

        return (nextMinDistance)*max(successorGameState.getScore(), 0) / (sumFood * len(newFood.asList())**2)

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

    def getAction(self, gameState):
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

        def rec_minmax(state, idx, depth):
            if idx == state.getNumAgents():
                idx = 0
                depth += 1

            if depth > self.depth or state.isWin() or state.isLose():
                return (self.evaluationFunction(state), [], state.isWin())

            vals = []
            for action in state.getLegalActions(idx):
                new_state = state.generateSuccessor(idx, action)
                val = rec_minmax(new_state, idx+1, depth)
                vals.append((val[0], [action] + val[1], val[2]))

            if idx == 0:
                return max(vals, key=lambda x: x[0])
            return min(vals, key=lambda x: x[0])

        return rec_minmax(gameState, 0, 1)[1][0]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def rec_minmax(state, idx, depth, maxVal, minVal):
            if idx == state.getNumAgents():
                idx = 0
                depth += 1

            v = 0
            if idx == 0:
                v = [float("-inf"), [], False]
            else:
                v = [float("inf"), [], False]

            if depth > self.depth or state.isWin() or state.isLose():
                return [self.evaluationFunction(state), [], state.isWin()]

            for action in state.getLegalActions(idx):
                new_state = state.generateSuccessor(idx, action)
                val = rec_minmax(new_state, idx+1, depth, maxVal, minVal)
                val[1] = [action]
                if idx == 0:
                    v = max([v, val], key=lambda x: x[0])
                    if v[0] > minVal:
                        return v
                    maxVal = max(maxVal, v[0])
                else:
                    v = min([v, val], key=lambda x: x[0])
                    if v[0] < maxVal:
                        return v
                    minVal = min(minVal, v[0])

            return v

        return rec_minmax(gameState, 0, 1, float("-inf"), float("inf"))[1][0]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def rec_expmax(state, idx, depth):
            if idx == state.getNumAgents():
                idx = 0
                depth += 1

            if depth > self.depth or state.isWin() or state.isLose():
                return (self.evaluationFunction(state), [], state.isWin())

            vals = []
            for action in state.getLegalActions(idx):
                new_state = state.generateSuccessor(idx, action)
                val = rec_expmax(new_state, idx+1, depth)
                vals.append((val[0], [action] + val[1], val[2]))

            if idx == 0:
                return max(vals, key=lambda x: x[0])
            newVal = sum([val[0] for val in vals])/len(vals)
            return (newVal, [], True)

        return rec_expmax(gameState, 0, 1)[1][0]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newGhostPos = [ghostState.getPosition() for ghostState in newGhostStates]
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    GhostStates = currentGameState.getGhostStates()
    ghostPos = [ghostState.getPosition() for ghostState in GhostStates]

    nextMinDistance = min(list(map(lambda x: abs(newPos[0] - x[0]) + abs(newPos[1] - x[1]), newGhostPos)))
    if len(newFood.asList()) == 0:
        return 1000000000000
    nextMinFood = min(list(map(lambda x: abs(newPos[0] - x[0]) + abs(newPos[1] - x[1]), newFood.asList())))
    sumFood = sum(list(map(lambda x: abs(newPos[0] - x[0]) + abs(newPos[1] - x[1]), newFood.asList())))

    if sum(newScaredTimes) > 0:
        scaredStates =  list(filter(lambda x: x.scaredTimer > 0, newGhostStates))
        scaredGhosts = [ghostState.getPosition() for ghostState in scaredStates]
        nextMinDistance = min(list(map(lambda x: abs(newPos[0] - x[0]) + abs(newPos[1] - x[1]), scaredGhosts)))
        notScaredStates = list(filter(lambda x: x not in scaredStates, newGhostStates))
        if len(notScaredStates) == 0:
            nextNotScaredMinDistance = 1
        else:
            nextNotScaredMinDistance = min(list(map(lambda x: abs(newPos[0] - x[0]) + abs(newPos[1] - x[1]), [ghost.getPosition() for ghost in notScaredStates])))
        if nextMinDistance == 0:
            return 1000000000
        return (nextNotScaredMinDistance)*abs(currentGameState.getScore())**2 / (sumFood * len(newFood.asList()) * (nextMinDistance+1))

    return (nextMinDistance)*max(currentGameState.getScore(), 0) / (sumFood * len(newFood.asList())**2)
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

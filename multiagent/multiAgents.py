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
import random
import util
import math

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # print(successorGameState)
        '''
        %%%%%%%%%%%%%%%%%%%%%%%%%
        %..    ....      ....   %
        %..       ...  ...  ... %
        %..       ...  ...  ... %
        %..     ...      ....   %
        %..       .>   ...  ... %
        %..         .  ...G ... %
        %..              ....  o%
        %%%%%%%%%%%%%%%%%%%%%%%%%
        '''
        curFood = currentGameState.getFood()
        # print(curFood)
        '''
        FTTFFFFFFFFFFFFFFTTTTFFFF
        FTTFFFFFFFFFFFFTTTFFTTTFF
        FTTFFFFFFFFFFFFTTTFFTTTFF
        FTTFFFFFFFFFFFFFFTTTTFFFF
        FFFFFFFFFFFFFFFFFFFFFFFFF
        '''
        food_list = curFood.asList()
        # print(food_list)
        '''
        5), (16, 6), (17, 1), (17, 2), (17, 3), (17, 4), (17, 5), (17, 6), (17, 7), (18, 1), (18, 4), (18, 7), (19, 1), (19, 4), (19, 7), (20, 1), (20, 2), (20, 3), (20, 4), (20, 5), (20, 6), (20, 7), (21, 2), (21, 3), (21, 5), (21, 6), (22, 2), (22, 3), (22, 5), (22, 6)]'''
        newPos = successorGameState.getPacmanPosition()  # (7, 5)
        # print(newPos)
        newFood = successorGameState.getFood()  # FTTFFFFFFFFFFFFFFTTTTFFFF
        # Ghost: (x,y)=(17.0, 3.0), South
        newGhostStates = successorGameState.getGhostStates()
        # print(newGhostStates[0])
        # print(type(newGhostStates))
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]
        # print(newScaredTimes)
        new_food_list = newFood.asList()
        # [(7, 5), (7, 5)]
        # [(17.0, 1.0)]
        ghosts_positions = successorGameState.getGhostPositions()
        # print(ghosts_positions)

        min_distance_to_ghost = float("inf")  # min distance to ghost
        for ghost in ghosts_positions:
            min_distance_to_ghost = min(manhattanDistance(
                ghost, newPos), min_distance_to_ghost)

        min_distance_to_food = float("inf")  # min distance to food
        for food in new_food_list:
            min_distance_to_food = min(manhattanDistance(
                food, newPos), min_distance_to_food)

        count = len(new_food_list)  # Number of food left

        if len(new_food_list) < len(food_list):  # If we eat food
            count = 10000  # count becomes very large
        if count == 0:  # If we eat all food
            count = -1000

        if min_distance_to_ghost < 2:  # If we are close to ghost
            min_distance_to_ghost = -100000
        else:
            min_distance_to_ghost = 0

        # print(newScaredTimes[0])
        if newScaredTimes[0] > 0:
            min_distance_to_ghost = 0
        # print(successorGameState.getScore())
        return (min_distance_to_ghost + 1.0 / min_distance_to_food + count) - successorGameState.getScore()

        return successorGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        """
        "*** YOUR CODE HERE ***"

        def minimax(state, depth, agent):

            # If we found the bottom nodes or we don't have any moves or we won or lost: Call evaluation function and return the result
            if depth == self.depth or state.getLegalActions(agent) == 0 or state.isWin() or state.isLose():
                return (self.evaluationFunction(state), None)

            minfinity = float("-inf")
            val = minfinity
            # If agent is pacman
            if (agent is 0):
                for a in state.getLegalActions(agent):
                    (v1, a1) = minimax(state.generateSuccessor(agent, a),
                                       depth, (agent + 1) % state.getNumAgents())
                    # Find the maximum value
                    if (v1 > val):
                        val = v1
                        maxa = a
            # Return the value and the action from which we found max
            if val is not minfinity:
                return (val, maxa)

            infinity = float("inf")
            val = infinity
            # If agent is ghost
            if (agent is not 0):
                for a in state.getLegalActions(agent):
                    # If it isn't the last ghost keep the same depth
                    if (((agent + 1) % state.getNumAgents()) is not 0):
                        (v1, a1) = minimax(state.generateSuccessor(agent, a),
                                           depth, (agent + 1) % state.getNumAgents())
                    # else if it is next_depth = depth + 1
                    else:
                        (v1, a1) = minimax(state.generateSuccessor(agent, a),
                                           depth + 1, (agent + 1) % state.getNumAgents())
                    # Find the minimum value
                    if (v1 < val):
                        val = v1
                        mina = a
            # Return the value and the action from which we found min
            if val is not infinity:
                return (val, mina)

      ############################################################
        return minimax(gameState, 0, 0)[1]
        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alpha_beta(state, depth, agent, A, B):

            # If we found the bottom nodes or we don't have any moves or we won or lost: Call evaluation function and return the result
            if depth == self.depth or state.getLegalActions(agent) == 0 or state.isWin() or state.isLose():
                return (self.evaluationFunction(state), None)

            minfinity = float("-inf")
            val = minfinity
            # If agent is pacman
            if (agent is 0):
                for a in state.getLegalActions(agent):
                    (v1, a1) = alpha_beta(state.generateSuccessor(agent, a),
                                          depth, (agent + 1) % state.getNumAgents(), A, B)

                    # Find the maximum value
                    if (v1 > val):
                        val = v1
                        maxa = a

                    if val > B:
                        return (val, maxa)
                    A = max(A, val)

            # Return the value and the action from which we found max
            if val is not minfinity:
                return (val, maxa)

            infinity = float("inf")
            val = infinity
            # If agent is ghost
            if (agent is not 0):
                for a in state.getLegalActions(agent):
                    # If it isn't the last ghost keep the same depth
                    if (((agent + 1) % state.getNumAgents()) is not 0):
                        (v1, a1) = alpha_beta(state.generateSuccessor(agent, a),
                                              depth, (agent + 1) % state.getNumAgents(), A, B)
                    # else if it is next_depth = depth + 1
                    else:
                        (v1, a1) = alpha_beta(state.generateSuccessor(agent, a),
                                              depth + 1, (agent + 1) % state.getNumAgents(), A, B)

                    # Find the minimum value
                    if (v1 < val):
                        val = v1
                        mina = a

                    if val < A:
                        return (val, mina)
                    B = min(B, val)

            # Return the value and the action from which we found min
            if val is not infinity:
                return (val, mina)

      ############################################################

        return alpha_beta(gameState, 0, 0, float("-inf"), float("inf"))[1]
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
        def expectimax(state, depth, agent):

            # If we found the bottom nodes or we don't have any moves or we won or lost: Call evaluation function and return the result
            if depth == self.depth or state.getLegalActions(agent) == 0 or state.isWin() or state.isLose():
                return (self.evaluationFunction(state), None)

            minfinity = float("-inf")
            val = minfinity
            # If agent is pacman
            if (agent is 0):
                for a in state.getLegalActions(agent):
                    (v1, a1) = expectimax(state.generateSuccessor(
                        agent, a), depth, (agent + 1) % state.getNumAgents())
                    # Find the maximum value
                    if (v1 > val):
                        val = v1
                        maxa = a
            # Return the value and the action from which we found max
            if val is not minfinity:
                return (val, maxa)

            infinity = float("inf")
            val = 0.0
            count = 0.0
            # If agent is ghost
            if (agent is not 0):
                for a in state.getLegalActions(agent):
                    # If it isn't the last ghost keep the same depth
                    if (((agent + 1) % state.getNumAgents()) is not 0):
                        (v1, a1) = expectimax(state.generateSuccessor(
                            agent, a), depth, (agent + 1) % state.getNumAgents())
                    # else if it is next_depth = depth + 1
                    else:
                        (v1, a1) = expectimax(state.generateSuccessor(
                            agent, a), depth + 1, (agent + 1) % state.getNumAgents())

                    # Find the average of the values
                    val += v1
                    count += 1
                    mina = a
            if val is not infinity:
                return (val/count, mina)

      ############################################################
        return expectimax(gameState, 0, 0)[1]
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
    #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Get the current score of the successor state
    score = currentGameState.getScore()

    ghostValue = 10.0
    foodValue = 10.0
    # bigger value for the scared ghost because we want to prefer it as a move
    scaredGhostValue = 50.0

    # For every ghost
    for x in newGhostStates:
        # Find the distance from pacman
        dis = manhattanDistance(newPos, x.getPosition())
        if dis > 0:
            """
            If the ghost is edible, and the ghost is near, the distance
            is small.In order to get a bigger score we divide the distance to a big number
            to get a higher score
            """
            if x.scaredTimer > 0:
                score += scaredGhostValue / dis
            else:
                score -= ghostValue / dis
            """
            If the ghost is not edible, and the ghost is far, the distance
            is big. We want to avoid such situation so we subtract the distance to a big number
            to lower the score and avoid this state.
            """

    # Find the distance of every food and insert it in a list using manhattan
    foodList = newFood.asList()
    foodDistances = []
    """
    If the food is very close to the pacman then the distance is small and
    we want such a situation to proceed. So we divide the distance to a big number
    to get a higher score
    """
    for x in foodList:
        foodDistances.append(manhattanDistance(newPos, x))

    # If there is at least one food
    if len(foodDistances) is not 0:
        score += foodValue / min(foodDistances)

    # Return the final Score
    return score

    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction

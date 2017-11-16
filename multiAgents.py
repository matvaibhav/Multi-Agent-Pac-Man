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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        noOfFood = successorGameState.getNumFood();

        minDistToGhost = 99999999
        for ghostState in newGhostStates:
            distGhost = manhattanDistance(newPos, ghostState.getPosition())
            if distGhost < minDistToGhost and distGhost > 0:
                minDistToGhost = distGhost

        minDistToFood = 99999999
        distancesToFood = [manhattanDistance(newPos, x) for x in newFood.asList()]
        if len(distancesToFood):
            minDistToFood = min(distancesToFood)

        return successorGameState.getScore() + (5.0 / minDistToFood) - (10.0 / minDistToGhost);


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
        return self.minimax(True, self.depth, gameState, 0)[1]

    """
        :param maxAgent: True if agent is pacman.
        :param depth: Depth till pacman agent will search for depth limited search.
        :param gameState: gameState of pacman for existing position.
        :param agentIndex: Index of agent. 0 for pacman 1 for minagent1 2 for minagent 2 and so on...
        :return : path to goal from source avoiding adversaries.

        In minimax search, pacman and the min player plays their best moves. Pacman tries for maximize
        his score by selection maximum from min nodes and adversaries tries to minimize pacman's score
        by selecting minimum values.

        In following code min and max players moves is handled in same function using recursion.
        We get the legal moves for current agent, get the successor game state and pass that state of 
        game to the next agent to play. Min agent calculates the min values as explained in algorithm
        and retrun it to the max and max agent maximises that values. Pacman decision is made on the basis
        of max score in particular direction.

    """

    def minimax(self, maxAgent, depth, gameState, agentIndex):
        actions = gameState.getLegalActions(agentIndex)
        if not actions or depth == 0 or gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP;
        if maxAgent:
            bestPossibleValue = -999999999;
            dir = Directions.STOP;
            for action in actions:
                successor = gameState.generateSuccessor(0, action);
                val = self.minimax(False, depth, successor, 1)[0]
                if val > bestPossibleValue:
                    bestPossibleValue = val
                    dir = action
            return bestPossibleValue, dir
        else:
            bestPossibleValue = 999999999
            dir = Directions.STOP;
            for action in actions:
                val = 0
                successor = gameState.generateSuccessor(agentIndex, action);
                if agentIndex == gameState.getNumAgents() - 1: # End of min Agents
                    val = self.minimax(True, depth - 1, successor, 0)[0]
                else:
                    val = self.minimax(False, depth, successor, agentIndex + 1)[0]

                if val < bestPossibleValue:
                    bestPossibleValue = val
                    dir = action
            return bestPossibleValue, dir


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alphaBeta(True, self.depth, gameState, 0, -999999999, 999999999)[1]

    """
        :param maxAgent: True if agent is pacman.
        :param depth: Depth till pacman agent will search for depth limited search.
        :param gameState: gameState of pacman for existing position.
        :param agentIndex: Index of agent. 0 for pacman 1 for minagent1 2 for minagent 2 and so on...
        :param alpha: best result for max agent along current path from root.
        :param beta: best result for min agent along current path from root.
        :return : path to goal from source avoiding adversaries.

        In alpha beta search, pacman and the min player plays their best moves. Pacman tries for maximize
        his score by selection maximum from min nodes and adversaries tries to minimize pacman's score
        by selecting minimum values. The addition here is we prune the tree by using alpha and beta values.
        So here we don't bother about the nodes which are not required to be explored on the basis of
        alpha and beta values and we prune them.

        In following code min and max players moves is handled in same function using recursion.
        We get the legal moves for current agent, get the successor game state and pass that state of 
        game to the next agent to play. Min agent calculates the min values as explained in algorithm
        and retrun it to the max and max agent maximises that values. Pacman decision is made on the basis
        of max score in particular direction. While deciding min max values here we return the bestvalue (i.e
        prune tree) if the value is below certain limit based on alpha beta.

    """

    def alphaBeta(self, maxAgent, depth, gameState, agentIndex, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        if not actions or depth == 0 or gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP;
        if maxAgent:
            bestPossibleValue = -999999999;
            dir = Directions.STOP;
            for action in actions:
                successor = gameState.generateSuccessor(0, action);
                val = self.alphaBeta(False, depth, successor, 1, alpha, beta)[0]
                if val > bestPossibleValue:
                    bestPossibleValue = val
                    dir = action
                # At max node (same as Pacaman's turn) if best current successor cost > beta stop further processing. A short circuit.
                if bestPossibleValue > beta:
                    return bestPossibleValue, dir
                alpha = max(alpha, bestPossibleValue)
            return bestPossibleValue, dir
        else:
            bestPossibleValue = 999999999
            dir = Directions.STOP;
            for action in actions:
                val = 0
                successor = gameState.generateSuccessor(agentIndex, action);
                if agentIndex == gameState.getNumAgents() - 1:
                    val = self.alphaBeta(True, depth - 1, successor, 0, alpha, beta)[0]
                else:
                    val = self.alphaBeta(False, depth, successor, agentIndex + 1, alpha, beta)[0]
                # At min node (same as a Ghost's turn) if best current successor cost < alpha stop further processing. A short circuit.
                if val < bestPossibleValue:
                    bestPossibleValue = val
                    dir = action

                if bestPossibleValue < alpha:
                    return bestPossibleValue, dir

                beta = min(beta, bestPossibleValue)

            return bestPossibleValue, dir


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
        return self.expectiMax(True, self.depth, gameState, 0)[1]

    """
        :param maxAgent: True if agent is pacman.
        :param depth: Depth till pacman agent will search for depth limited search.
        :param gameState: gameState of pacman for existing position.
        :param agentIndex: Index of agent. 0 for pacman 1 for minagent1 2 for minagent 2 and so on...
        :return : path to goal from source avoiding adversaries.

        In expectiMax search, pacman does not consider min as best player. Pacman tries for maximize
        his score by selection maximum from min nodes and adversaries tries to minimize pacman's score
        by selecting minimum values. This is also same as minimax the difference is as pacman takes average
        of minimum values to choose best moves.

        In following code min and max players moves is handled in same function using recursion.
        We get the legal moves for current agent, get the successor game state and pass that state of 
        game to the next agent to play. Min agent calculates the average of min values as explained in algorithm
        and retrun it to the max and max agent maximises that values. Pacman decision is made on the basis
        of max score in particular direction.

    """

    def expectiMax(self, maxAgent, depth, gameState, agentIndex):
        actions = gameState.getLegalActions(agentIndex)
        if not actions or depth == 0 or gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP;
        if maxAgent:
            bestPossibleValue = -999999999;
            dir = Directions.STOP;
            for action in actions:
                successor = gameState.generateSuccessor(0, action);
                val = self.expectiMax(False, depth, successor, 1)[0]
                if val > bestPossibleValue:
                    bestPossibleValue = val
                    dir = action
            return bestPossibleValue, dir
        else:
            bestPossibleValue = 999999999
            dir = Directions.STOP;
            valList = []
            for action in actions:
                val = 0
                successor = gameState.generateSuccessor(agentIndex, action);
                if agentIndex == gameState.getNumAgents() - 1:
                    val = self.expectiMax(True, depth - 1, successor, 0)[0]
                else:
                    val = self.expectiMax(False, depth, successor, agentIndex + 1)[0]

                valList.append(val)
            return sum(valList) / float(len(valList)), None


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction


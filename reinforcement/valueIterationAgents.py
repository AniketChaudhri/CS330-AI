# valueIterationAgents.py
# -----------------------
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


import collections
import mdp
import util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        * Q1 *

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()
        self.runValueIteration()

    def runValueIteration(self):
        '''
            v[k](s) = max[a] Q[k](s, a)
            Q[k+1](s, a) = sum[s'] T(s, a, s') * [r(s, a, s') + discount*v[k](s')]

        '''
        "*** YOUR CODE HERE ***"
        # * Q1
        # for each iteration, the value of each state is updated using the bellman equation
        for i in range(self.iterations):
            iterationValues = util.Counter()  # Counter to hold the values for each iteration
            for s in self.mdp.getStates():  # for each state, the value is calculated as the max of q-values for all possible actions
                # if the state is terminal, the value is 0
                if self.mdp.isTerminal(s):
                    self.values[s] = self.mdp.getReward(
                        s, 'exit', '')  # reward for terminal state is 0
                else:
                    # getting all possible actions for a state
                    actions = self.mdp.getPossibleActions(s)
                    iterationValues[s] = max(
                        [self.computeQValueFromValues(s, a) for a in actions])  # getting the max q-value for a state
            self.values = iterationValues   # updating the values for the next iteration

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(
            state, action)  # getting the transition states and probabilities for a state action pair
        value = 0
        for ts in transitionStatesAndProbs:  # for each transition state, the q-value is calculated as the sum of the transition probability and the discounted value of the next state
            stateTransitionReward = self.mdp.getReward(state, action, ts[0])
            value = value + stateTransitionReward + \
                self.discount * \
                (self.values[ts[0]]*ts[1]
                 )  # q-value = reward + discount * value of next state
            # print value

        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # initializing a stateAction counter, which is used to hold the q-value for each
        # state action pair. The policy or action is the one that gives the best expected sum of rewards.
        stateAction = util.Counter()
        for a in self.mdp.getPossibleActions(state):
            stateAction[a] = self.computeQValueFromValues(state, a)
        policy = stateAction.argMax()
        return policy

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    # * Q4
    """
        * Please read learningAgents.py before reading this.*
        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        states = mdp.getStates()

        for k in range(self.iterations):
            state = states[k % len(states)]
            possibleActions = mdp.getPossibleActions(state)
            if not possibleActions:  # if the state is terminal, the value is 0
                continue
            else:   # if the state is not terminal, the value is calculated as the max of q-values for all possible actions
                qValues = [self.computeQValueFromValues(state, action)
                           for action in possibleActions]
                self.values[state] = max(qValues)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}  # a dictionary to hold the predecessors of each state

        for state in self.mdp.getStates():  # for each state, the predecessors are calculated
            # if the state is not terminal, the predecessors are calculated
            if not self.mdp.isTerminal(state):
                # initializing the max value to -infinity
                maxVal = -float('inf')
                # for each action, the q-value is calculated
                for action in self.mdp.getPossibleActions(state):
                    # for each transition state, the q-value is calculated
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):

                        if nextState in predecessors:
                            # adding the state to the predecessors of the transition state
                            predecessors[nextState].add(state)
                        else:
                            # adding the state to the predecessors of the transition state
                            predecessors[nextState] = {state}

        pq = util.PriorityQueue()   # initializing a priority queue

        for s in self.mdp.getStates():  # for each state, the priority is calculated and added to the priority queue
            # if the state is not terminal, the priority is calculated
            if not self.mdp.isTerminal(s):
                maxQ = -float('inf')

                for action in self.mdp.getPossibleActions(s):
                    Q = self.computeQValueFromValues(s, action)

                    if Q > maxQ:
                        maxQ = Q
                diff = abs(maxQ - self.values[s])

                pq.update(s, -diff)

        # for each iteration, the state with the highest priority is popped from the priority queue
        for _ in range(self.iterations):
            if pq.isEmpty():    # if the priority queue is empty, the iteration is stopped
                break

            s = pq.pop()    # the state with the highest priority is popped from the priority queue

            if not self.mdp.isTerminal(s):
                maxVal = -float('inf')

                for action in self.mdp.getPossibleActions(s):
                    Q = self.computeQValueFromValues(s, action)
                    if Q > maxVal:
                        maxVal = Q

                self.values[s] = maxVal

            # for each predecessor of the state, the priority is calculated and added to the priority queue
            for p in predecessors[s]:
                maxQ = -float('inf')

                for action in self.mdp.getPossibleActions(p):
                    Q = self.computeQValueFromValues(p, action)

                    if Q > maxQ:
                        maxQ = Q
                d = abs(maxQ - self.values[p])

                if d > self.theta:
                    pq.update(p, -d)

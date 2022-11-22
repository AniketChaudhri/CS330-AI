# python code to calculate input probability query for a bayesian network given in a file in the format 
# No. of random variables, list of random variables
# Random variables and possible values
# Conditional probability tables
# Query probability
# Input: bayesnet.txt
# Output: probability of query

import sys
import math
import itertools
from collections import defaultdict
import random
from copy import deepcopy, copy

# function to generate samples from the bayesian network
def generateSamples(bayesNet, numSamples):
    samples = []
    for i in range(0, numSamples):
        sample = {}
        for j in range(0, len(bayesNet.vars)):
            var = bayesNet.vars[j]
            # get the conditional probability table for the variable
            cpt = bayesNet.getCpt[var]
            # get the conditioned variables
            conditionedVars = bayesNet.parents[var]
            # get the values of the conditioned variables
            conditionedValues = []
            for k in range(0, len(conditionedVars)):
                conditionedValues.append(sample[conditionedVars[k]])
            # get the probability of the variable given the conditioned variables
            prob = cpt[tuple(conditionedValues)]
            # generate a random number
            rand = random.random()
            # if the random number is less than the probability, set the variable to yes, else no
            if rand < prob:
                sample[var] = 'yes'
            else:
                sample[var] = 'no'
        samples.append(sample)
    return samples












class BayesNet(object):

    def __init__(self, variables, inputInEdges, inputOutEdges, inputVariableDomains):
        """
        Bare bones constructor for bayes nets.
        Use constructEmptyBayesNet for a nicer interface.

        variables:       An iterable of all of the variables.
        inEdges:         A dict that maps variable to otherVariable
                         when there is an edge from otherVariable to
                         variable 
        outEdges:        A dict that maps variable to otherVariable
                         where there is an edge from variable to
                         otherVariable  
        variableDomains: A dict mapping each variable to its domain (list like).

        Constructs a bayes net with edges given by inEdges and
        outEdges for each variable.
        Doesn't initialize the conditional probability table for any variables.
        """
        # Each variable is unique (so that they can be keys in dicts)
        self.__variablesSet = set(variables)
        self.__variables = sorted(list(variables))
        # self.__inEdges[v] = [u if the edge (u, v) exists]
        self.__inEdges  = inputInEdges 
        # self.__outEdges[u] = [v if the edge (u, v) exists]
        self.__outEdges = inputOutEdges

        # make sure that the edge maps contain all variables
        for variable in self.__variablesSet:
            if variable not in self.__inEdges:
                self.__inEdges[variable]  = set()
            if variable not in self.__outEdges:
                self.__outEdges[variable] = set()

        self.__variableDomainsDict = inputVariableDomains
        self.__CPTDict = {}

    def variablesSet(self):
        " Returns a copy of the set of variables in the bayes net "
        return copy(self.__variablesSet)

    def variableDomainsDict(self):
        " Returns a copy of the variable domains in the bayes net "
        return deepcopy(self.__variableDomainsDict)

    def inEdges(self):
        " Returns a copy of the incoming edges in the bayes net "
        return deepcopy(self.__inEdges)

    def outEdges(self):
        " Returns a copy of the outgoing edges in the bayes net "
        return deepcopy(self.__outEdges)

    def __str__(self):
        """
        Human-readable representation of a bayes net. 
        Prints each variable, each edge, and then each conditional probability table.
        """
        netString = "Variables: " + ", ".join([str(var) for var in self.__variablesSet]) + "\n" + \
                    "Edges: " + ", ".join([str(fromVar) + " -> " + str(toVar) \
                    for toVar in self.__variablesSet \
                    for fromVar in self.__inEdges[toVar]])
        try:
            factorsString = "Conditional Probability Tables:\n\n" + \
                            "\n ======================= \n\n".join([str(factor) for factor in self.getAllCPTsWithEvidence()])
            return netString + '\n\n' + factorsString
        except KeyError:
            return netString + '\n' + repr(self.variableDomainsDict())

    def sameGraph(self, other):
        sameVars = self.__variables == other.__variables
        sameInEdges = sorted(self.__inEdges) == sorted(other.__inEdges)
        sameOutEdges = sorted(self.__outEdges) == sorted(other.__outEdges)
        return sameVars and sameInEdges and sameOutEdges

    def linearizeVariables(self):
        """
        Returns a list of the variables in the bayes net, but in a
        linearized order (parents of a variable in the graph always
        precede it in the order).

        Useful for sampling.
        """
        inEdgesIncremental = dict([(var, edgeSet.copy()) for (var, edgeSet) in self.__inEdges.items()])
        noIncomingList = [var for var in self.__variables if len(self.__inEdges[var]) == 0]

        linearizedList = []

        while len(noIncomingList) > 0:
            currentVar = noIncomingList.pop()
            linearizedList.append(currentVar)
            for outGoingVariable in self.__outEdges[currentVar]:
                inEdgesIncremental[outGoingVariable].remove(currentVar)
                if len(inEdgesIncremental[outGoingVariable]) == 0:
                    noIncomingList.append(outGoingVariable)

        hasEdgesLeftOver = any([len(inEdgesIncremental[var]) > 0 for var in self.__variables])
        if hasEdgesLeftOver:
            raise ValueError("Graph has at least one cycle (not a bayes net) " + \
                               str(inEdgesIncremental))
        else:
            return linearizedList

    def getCPT(self, variable):
        """
        Returns a copy of the conditional probability table in the bayes net
        for variable.  This is instantiated as a factor.
        """
        if variable not in self.__variablesSet:
            raise ValueError("Variable not in bayes net: " + str(variable))
        else:
            return deepcopy(self.__CPTDict[variable])

    def setCPT(self, variable, CPT):
        """
        Sets the conditional probability table in the bayes net for
        variable.

        CPT is a Factor of the conditional probability table with variable 
        as the only unconditioned variable, and each conditioned variable
        must have an edge going into variable.
        """
        if variable not in self.__variablesSet:
            raise ValueError("Variable not in bayes net: " + str(variable))
        else:
            unconditionedVariables = CPT.unconditionedVariables()
            conditionedVariables = CPT.conditionedVariables()

            if len(unconditionedVariables) != 1:
                raise ValueError("Unconditioned variables must contain a single element for an entry" + \
                                   " in the conditional probability tables for this Bayes net\n" + \
                                  str(unconditionedVariables))


            unconditionedVariable = list(unconditionedVariables)[0]
            
            if unconditionedVariable != variable:
                raise ValueError("Variable in the input and the " 
                                  + "unconditionedVariable in the factor must \nagree. " +
                                  "Input variable: " + str(variable) + \
                                  " unconditioned variable: " + str(unconditionedVariable))

            for var in conditionedVariables:
                if var not in self.__inEdges[unconditionedVariable]:
                    raise ValueError("Conditioned variables must be all have an edge " +
                                       "going into \n the unconditionedVariable. \n" +
                                       "conditionedVariables: " + str(conditionedVariables) + \
                                       "\nparent: " + str(var))

            self.__CPTDict[variable] = deepcopy(CPT)

    def getReducedVariableDomains(self, evidenceDict):
        """
        evidenceDict: A dict with an assignment for each
                                evidence variable.

        Returns a new variableDomainsDict where each evidence
        variable's domain is the single value that it is being
        assigned to (and is otherwise unchanged).
        """
        reducedVariableDomainsDict = self.variableDomainsDict()
        for (evidenceVariable, value) in evidenceDict.items():
            reducedVariableDomainsDict[evidenceVariable] = [value]
        return reducedVariableDomainsDict

    def getCPTWithEvidence(self, variable, evidenceDict=None):
        """
        Gets a conditional probability table for a variable, where the
        assignments in evidenceDict have been performed, so
        the CPT may have less rows than what
        would be returned from getCPT.

        Input evidenceDict is optional.
        If it is not provided, the CPTs for all variables without 
        specializing the domains is provided.
        """
        if evidenceDict is None or len(evidenceDict.items()) == 0:
            return self.getCPT(variable)
        else:
            reducedVariableDomains = self.getReducedVariableDomains(evidenceDict)
            variableCPT = self.getCPT(variable)
            return variableCPT.specializeVariableDomains(reducedVariableDomains)

    def getAllCPTsWithEvidence(self, evidenceDict=None):
        """
        Returns a list of conditional probability tables (taking into
        account evidence) for all variables in the bayes net.

        Input evidenceDict is optional.
        If it is not provided, the CPTs for all variables without 
        specializing the domains is provided.
        """
        return [self.getCPTWithEvidence(var, evidenceDict) for var in self.__variablesSet]

    def easierToParseString(self, printVariableDomainsDict=False):
        " Used internally for computer-readable printing "
        returnStrings = []
        for CPT in self.getAllCPTsWithEvidence():
            # CPT has only one unconditioned variable, extract it and use as a prefix
            prefix = next(iter(CPT.unconditionedVariables()))
            returnStrings.append(CPT.easierToParseString(prefix=prefix, printVariableDomainsDict=printVariableDomainsDict))
            printVariableDomainsDict = False
        return "\n".join(returnStrings)


# function to read the bayesian network from the file
def readBayesNet(filename):
    f = open(filename, 'r')
    # first line contains number of random variables and list of random variables
    line = f.readline()
    line = line.strip()
    tokens = line.split(', ')
    numVars = int(tokens[0])
    vars = tokens[1:]
    # second line contains random variables and possible values
    varValues = {}
    for i in range(0, len(vars)):
        line = f.readline()
        line = line.strip()
        tokens = line.split(', ')
        varValues[tokens[0]] = tokens[1:]
    # third line contains conditional probability tables
    edgeTupleList = []
    cpt = {}
    for i in range(0, len(vars)):
        line = f.readline()
        line = line.strip()
        # get conditioned and unconditioned variables
        tokens = line.split(' | ')
        if len(tokens) == 1:
            conditioned = []
            unconditioned = tokens[0].split(', ')
        else:
            conditioned = tokens[0].split(', ')
            unconditioned = tokens[1].split(', ')

            # add edges from conditioned to unconditioned
            for j in range(0, len(conditioned)):
                edgeTupleList.append((conditioned[j], unconditioned[0]))

        # get conditional probability table
        cpt[unconditioned[0]] = {}
        # number of rows in the table
        numRows = len(varValues[unconditioned[0]])
        for j in range(0, len(conditioned)):
            numRows = numRows * len(varValues[conditioned[j]])

        for j in range(0, numRows):
            line = f.readline()
            line = line.strip()
            tokens = line.split(', ')
            # in each row, unconditioned variable is second last and probability is last, rest are conditioned variables
            prob = float(tokens[len(tokens) - 1])
            unconditionedValue = tokens[len(tokens) - 2]
            conditionedValues = tokens[0:len(tokens) - 2]
            # add to the conditional probability table
            # save in format unconditionedVariable : [{for each variable: yes/no}, probability]
            if len(conditioned) == 0:
                cpt[unconditioned[0]][()] = prob
            cpt[unconditioned[0]][tuple(tokens[:-1])] = float(tokens[-1])

    # create the bayesian network
    bayesNet = BayesNet(vars, edgeTupleList, varValues)

    line = f.readline()
    line = line.strip()
    tokens = line.split(': ')
    queryVar = tokens[1]
    # segregate conditioned and unconditioned variables from given query
    vars = queryVar.split('|')
    if len(vars) == 1:
        queryUnconditioned = vars[0].split(', ')
        queryConditioned = []
    else:
        queryUnconditioned = vars[1].split(', ')
        queryConditioned = vars[0].split(', ')

    return (bayesNet, queryUnconditioned, queryConditioned)


if __name__ == '__main__':

    # check if input is taken properly
    if len(sys.argv) != 2:
        print ('Usage: python bayesnetprob.py bayesnet.txt')
        sys.exit(1)

    # read the bayesian network from the file
    (bayesNet, queryUnconditioned, queryConditioned) = readBayesNet(sys.argv[1])
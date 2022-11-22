'''
Author: 
1. Aniket Akshay Chaudhri (2003104)
2. Adarsh Anand(2003101)

Bayes Net Probability Calculator

Instructions:
1. Run the program using the command: python3 bnp.py template.txt
2. Maintain the format of the template.txt file as given in the template.txt file.
'''

# importing the required modules
import numpy as np
import pandas as pd
from progress.bar import ChargingBar as Bar # progress bar
from random import randint
from collections import Counter


# class for bayes net
class BayesNet(object):

    # initialise the bayes net with name, nodes and conditional probability distributions
    def __init__(self, name):
        self.name = name
        self.node = dict()
        self.cpt = dict()
        return

    def add_node(self, name, levelsCount, values):
        '''function to add node to bayes net'''
        self.node[name] = {'n': levelsCount, 'levels': values}
        return

    def add_probability(self, child, parents, table):
        '''function to add CPT '''
        self.node[child]['parents'] = parents # add parents to node
        indices = list() 
        values = list()
        for row in table:
            indices.append(row[0])
            values.append(row[1])
        if len(parents) > 1:
            index = pd.MultiIndex.from_tuples(indices, names=parents)
            self.cpt[child] = pd.DataFrame(data=values, columns=self.node[child]['levels'], index=index)
            
        if len(parents) == 1 and parents[0] is not None:
            df = pd.DataFrame(data=values, columns=self.node[child]['levels'], index=indices)
            df.index.name = parents[0]
            self.cpt[child] = df
        if parents[0] is None:
            self.cpt[child] = pd.DataFrame(data=values, columns=self.node[child]['levels'])
        # print(self.cpt[child])
        return
    
    def get_probability(self, name, evidence=dict(), value=None):
        ''' function to obtain probability of a node given some of its parent evidence'''
        cpt = self.cpt[name]
        # if more than one evidence is given
        if len(evidence) > 1:
            levels = list(evidence.keys())
            values = list(evidence.values())
            if value is not None:
                # return probability of a particular value given evidence
                return cpt.xs(values, level=levels)[value]
            else:
                # return probability of all values given evidence
                return cpt.xs(values, level=levels)
        elif len(evidence) == 1:
            values = list(evidence.values())
            if value is not None:
                return cpt.loc[values[0], value]
            else:
                return cpt.loc[values[0]]
        elif len(evidence) == 0:
            if value is not None:
                return cpt[value]
            else:
                return cpt

    def sample(self, name, given=dict(), r=None):
        '''function to sample a node given some of its parent evidence'''
        parents = self.node[name]['parents']
        if parents[0] is None:
            # if no parents, sample from the prior distribution
            cpt = self.get_probability(name)
        else:
            # if parents, sample from the conditional distribution
            cpt = self.get_probability(name, evidence=given)
        if r is None: # if no random number is given, generate one
            r = np.random.uniform()
        sum = 0 # sum of probabilities
        if len(cpt.shape) == 1: # if only one value is present
            cpt = dict(cpt) # convert to dictionary
            for key in cpt: 
                sum += cpt[key] # add probability to sum
                if r < sum: # if random number is less than sum, return the value
                    return key
        if len(cpt.shape) == 2: # if more than one value is present
            for column in cpt: # iterate over columns of the dataframe 
                sum += list(cpt[column])[0] # add probability to sum 
                if r < sum: # if random number is less than sum, return the value
                    return column

    def compute_sample(self, preset=dict()):
        unobserved = set(self.node.keys())
        sample = dict(preset)
        unobserved = unobserved - set(sample.keys())
        for node in unobserved:
            sample = self.sample_nodes([node], sample)
            unobserved = unobserved - set(sample.keys())
        return sample

    def random_sample(self, preset=dict()):
        ''' function to generate a random sample'''
        unobserved = set(self.node.keys()) # unobserved nodes
        sample = dict(preset) # sample dictionary
        unobserved = unobserved - set(sample.keys()) # unobserved nodes
        for node in unobserved: # iterate over unobserved nodes
            values = bn.cpt[node] # get the conditional probability distribution
            values = list(values) # convert to list
            ind = randint(0, len(values)-1) # generate a random index
            sample[node] = values[ind] # add the value to the sample
        return sample

    def sample_nodes(self, nodes, sample):
        for node in nodes:
            if node not in sample.keys():
                parents = self.node[node]['parents']
                if parents[0] is None:
                    sample[node] = self.sample(node)
                elif set(parents) <= set(sample.keys()):
                    given = {key: sample[key] for key in parents}
                    sample[node] = self.sample(node, given)
                else:
                    sample = self.sample_nodes(nodes=parents, sample=sample)
                    given = {key: sample[key] for key in parents}
                    sample[node] = self.sample(node, given)
        return sample

    def prior_sample(self, predict=dict(), given=dict(), n=1000):
        ''' function to calculate probability using prior sampling'''
        bar = Bar('Sampling', max=n) # progress bar
        samples = list()
        for i in range(n):
            samples.append(self.compute_sample())
            bar.next()
        # find number of samples which match the evidence and number of samples which match both evidence and query
        positives = 0
        positives_and_query = 0
        for sample in samples:
            evidence = {key: sample[key] for key in given.keys()}
            if evidence == given:
                positives += 1
                for key in predict.keys():
                    if sample[key] == predict[key]:
                        positives_and_query += 1
        bar.finish()
        return positives_and_query/positives


    def rejection_sample(self, predict=dict(), given=dict(), n=10000):
        '''Byes Net Sampling using Rejection Sampling.
        Randomly computes a sample from the Bayes Net and checks if it satisfies the given evidence.
        If it does, it is added to the list of samples.'''

        sum = 0 # initialise number of positive samples
        count = 0
        bar = Bar('Sampling', max=n)
        while count < n:
            
            sample = self.compute_sample() # compute a sample

            # if some of evidence does not match the sample, reject the sample
            evidence = {key: sample[key] for key in given.keys()}
            if not given == evidence:
                continue
            evidence = {key: sample[key] for key in predict.keys()}
            if predict == evidence:
                sum += 1
            count += 1
            bar.next()

            
            # sum += 1 # increment number of positive samples
        bar.finish()
        return sum/n # return probability of positive samples

    def likelihood_weighting(self, predict=dict(), given=dict(), n=10000):
        num = den = 0
        bar = Bar('Sampling', max=n)
        for i in range(n):
            bar.next()
            sample = self.compute_sample(preset=predict)
            for node in predict.keys():
                parents = self.node[node]['parents']
                if (parents == [None]):
                    given_pa = dict()
                else:
                    given_pa = {key: sample[key] for key in parents}
                weight = float(self.get_probability(node, evidence=given_pa, value=predict[node]))
            evidence = {key: sample[key] for key in given.keys()}
            if given == evidence:
                num += weight
            den += weight
        bar.finish()
        return num/den

    def gibbs_sampling(self, predict=dict(), given=dict(), n=10000, skip = 50):
        bar = Bar('Sampling', max=n)
        nodes = list(self.node.keys())
        sample = self.random_sample(preset=given)
        count = 0
        sum = 0
        for i in range(n):
            last = None
            bar.next()
            node = None
            last = node
            while node is None or node in given.keys() or node == last:
                node = nodes[randint(0,len(nodes)-1)]
            parents = self.node[node]['parents']
            if parents[0] is None:
                sample[node] = self.sample(node)
            else:
                given = {key: sample[key] for key in parents}
                sample[node] = self.sample(node, given=given)
            if count == skip:
                evidence = {key: sample[key] for key in predict.keys()}
                if not predict == evidence:
                    continue
                sum += 1
            else:
                count += 1
        bar.finish()
        return sum/(n-count)

# create a bayes net object
bn = BayesNet('Bayes Net')

def createBayesNet(filename):   
    ''' function to create a bayes net from a file
    input: filename
    output: bayes net object, query node, evidence'''

    f = open(filename, 'r') # open the file
    line = f.readline()
    # first line contains the number of nodes and names of nodes
    line = line.split(', ')
    num_nodes = int(line[0])
    nodes = line[1:]
    # next num_nodes lines contain the node name and names of levels
    for i in range(num_nodes):
        line = f.readline()
        line = line.split(', ')
        node = line[0]
        levels = line[1:]
        # remove newline character from last level
        levels[-1] = levels[-1].strip()
        # print(levels)
        bn.add_node(node, len(levels), tuple(levels))
    # next num_nodes blocks contain CPTs
    for i in range(num_nodes):
        line = f.readline()
        line = line.split(' | ')
        if len(line) > 1: 
            node = line[0]
            parents = line[1].split(',')
            # strip off the newline character
            parents[-1] = parents[-1][:-1]
        else:
            node = line[0][0]
            parents = [None]
        # print(i, node)
        # print(i, parents)
        # compute number of rows
        rows = 1
        for parent in parents:
            if parent is not None:
                rows *= len(bn.node[parent]['levels'])
        valueList = []
        for j in range(rows):
            line = f.readline()
            line = line.split(', ')
            # remove newline character from last level
            line[-1] = line[-1].strip()
            values = tuple()
            if parents == [None]:
                values = None
            elif len(parents) == 1:
                values = line[0]
            else:
                for k in range(len(parents)):
                    values += (line[k],)

            temptuple = [] # create a temporary array to store the values of probability
            temptuple.append(float(line[-1]))
            for k in range(len(bn.node[node]['levels'])-1): # append the rest of the values
                line = f.readline()
                line = line.split(', ')
                temptuple.append(float(line[-1][:-1]))
            
            temptuple = tuple([value for value in temptuple])
            valueList.append((values, temptuple))
        # print(node, parents, valueList)
        bn.add_probability(node, parents, valueList)

    # get query of format Query: P( B=+b| J=+j)
    line = f.readline()
    line = line.split(': ')
    query = line[1]
    query = query[3:-1]
    query = query.split('| ')
    predict = query[0]
    # predict = predict.split(', ')
    predict = {predict[0]:predict[2:]}
    given = query[1]
    given = given.split(', ')
    givendict = {}
    for i in range(len(given)):
        givendict[given[i][0]] = given[i][2:]

    return bn, predict, givendict


def main():
    # get the bayes net object, query node and evidence
    bn, predict, given = createBayesNet('template.txt')

    # print(bn.prior_sample(predict=predict, given=given, n=10000))
    # print(bn.rejection_sample(predict=predict, given=given, n=1000))
    print(bn.likelihood_weighting(predict=predict, given=given, n=1000))
    # print(bn.gibbs_sampling(predict=predict, given=given, n=1000, skip=50))


if __name__ == '__main__':
    main()
    # test()

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
        self.cpd = dict()
        return

    # function to add node to bayes net
    def add_node(self, name, node_type, levels, values):
        self.node[name] = {'type': node_type, 'n': levels, 'levels': values}
        return

    def add_probability(self, child, parents, table):
        self.node[child]['parents'] = parents
        indices = list()
        values = list()
        for row in table:
            indices.append(row[0])
            values.append(row[1])
        if len(parents) > 1:
            index = pd.MultiIndex.from_tuples(indices, names=parents)
            self.cpd[child] = pd.DataFrame(data=values, columns=self.node[child]['levels'], index=index)
        if len(parents) == 1 and parents[0] is not None:
            df = pd.DataFrame(data=values, columns=self.node[child]['levels'], index=indices)
            df.index.name = parents[0]
            self.cpd[child] = df
        if parents[0] is None:
            self.cpd[child] = pd.DataFrame(data=values, columns=self.node[child]['levels'])
        return

    def get_probability(self, name, evidence=dict(), value=None):
        cpd = self.cpd[name]
        if len(evidence) > 1:
            levels = list(evidence.keys())
            values = list(evidence.values())
            if value is not None:
                return cpd.xs(values, level=levels)[value]
            else:
                return cpd.xs(values, level=levels)
        elif len(evidence) == 1:
            values = list(evidence.values())
            if value is not None:
                return cpd.loc[values[0], value]
            else:
                return cpd.loc[values[0]]
        elif len(evidence) == 0:
            if value is not None:
                return cpd[value]
            else:
                return cpd

    def sample(self, name, given=dict(), r=None):
        parents = self.node[name]['parents']
        if parents[0] is None:
            cpd = self.get_probability(name)
        else:
            if not set(parents) == set(given.keys()):
                raise Exception('Parent samples need to be provided to sample child.')
            cpd = self.get_probability(name, evidence=given)
        if r is None:
            r = np.random.uniform()
        sum = 0
        if len(cpd.shape) == 1:
            cpd = dict(cpd)
            for key in cpd:
                sum += cpd[key]
                if r < sum:
                    return key
        if len(cpd.shape) == 2:
            for column in cpd:
                sum += list(cpd[column])[0]
                if r < sum:
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
        unobserved = set(self.node.keys())
        sample = dict(preset)
        unobserved = unobserved - set(sample.keys())
        for node in unobserved:
            values = bn.cpd[node]
            values = list(values)
            ind = randint(0, len(values)-1)
            sample[node] = values[ind]
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

    def rejection_sample(self, predict=dict(), given=dict(), n=10000):
        sum = 0
        bar = Bar('Sampling', max=n)
        for i in range(n):
            bar.next()
            sample = self.compute_sample()
            evidence = {key: sample[key] for key in given.keys()}
            if not given == evidence:
                continue
            evidence = {key: sample[key] for key in predict.keys()}
            if not predict == evidence:
                continue
            sum += 1
        bar.finish()
        return sum/n

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

    def predict(self, nodes, algorithm='gibbs', given=dict(), n=10000, skip=100):
        p = dict()
        for node in nodes:
            p[node] = dict()
            levels = self.node[node]['levels']
            for level in levels:
                predict = {node:level}
                if algorithm == 'gibbs':
                    p[node][level] = self.gibbs_sampling(predict=predict, given=given, n=n, skip=skip)
                elif algorithm == 'reject':
                    p[node][level] = self.rejection_sample(predict=predict, given=given, n=n)
                elif algorithm == 'weighting':
                    p[node][level] = self.likelihood_weighting(predict=predict, given=given, n=n)
            p[node] = Counter(p[node]).most_common(1)
        return p


bn = BayesNet('Bayes Net')
def createBayesNet(filename):
    

    f = open(filename, 'r')
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
        bn.add_node(node, 'discrete', len(levels), tuple(levels))
    # next num_nodes blocks contain CPTs
    for i in range(num_nodes):
        line = f.readline()
        line = line.split(' | ')
        # print(line)
        # node = line[0]
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
            # if len(bn.node[node]['levels']) == 2:


            temptuple = []
            temptuple.append(float(line[-1]))
            for k in range(len(bn.node[node]['levels'])-1):
                line = f.readline()
                line = line.split(', ')

                temptuple.append(float(line[-1][:-1]))
            # if len(temptuple) == 1:
            #     valueList.append((values, temptuple[0]))
            # else:
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
    bn, predict, given = createBayesNet('example.txt')
    # print(predict, given)
    # print(bn.get_probability('J'))
    # print(bn.get_probability('J', evidence={'A': '-a'}))
    # prob = bn.likelihood_weighting(predict={'A': '+a'}, given={'B': '-b', 'E': '+e'}, n=1000)

    # prob = bn.rejection_sample(
    #     predict={'A': '+a'}, given={'B': '-b', 'E': '+e'}, n=10000)
    # print(prob)
    # prob = bn.gibbs_sampling(predict={'A': '+a'}, given={
    #                          'B': '-b', 'E': '+e'}, n=10000, skip=50)
    # print(bn.likelihood_weighting(predict={'B': '+b'}, given={'J': '+j', 'A': '-a'}, n=1000))
    print(bn.likelihood_weighting(predict=predict, given=given, n=1000))
    # print(prob)
    # print(bn.predict(nodes=['A'], algorithm='gibbs',
    #       given={'B': '+b', 'E': '-e'}, n=10000, skip=50))



def test():


    bn = Network('Alarm')
    bn.add_node('HISTORY', 'discrete', 2, ('TRUE', 'FALSE'))
    bn.add_node('CVP', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))
    bn.add_node('PCWP', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))
    bn.add_node('HYPOVOLEMIA', 'discrete', 2, ('TRUE', 'FALSE'))
    bn.add_node('LVEDVOLUME', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))
    bn.add_node('LVFAILURE', 'discrete', 2, ('TRUE', 'FALSE'))
    bn.add_node('STROKEVOLUME', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))
    bn.add_node('ERRLOWOUTPUT', 'discrete', 2, ('TRUE', 'FALSE'))
    bn.add_node('HRBP', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))
    bn.add_node('HREKG', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))
    bn.add_node('ERRCAUTER', 'discrete', 2, ('TRUE', 'FALSE'))
    bn.add_node('HRSAT', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))
    bn.add_node('INSUFFANESTH', 'discrete', 2, ('TRUE', 'FALSE'))
    bn.add_node('ANAPHYLAXIS', 'discrete', 2, ('TRUE', 'FALSE'))
    bn.add_node('TPR', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))
    bn.add_node('EXPCO2', 'discrete', 4, ('ZERO', 'LOW', 'NORMAL', 'HIGH'))
    bn.add_node('KINKEDTUBE', 'discrete', 2, ('TRUE', 'FALSE'))
    bn.add_node('MINVOL', 'discrete', 4, ('ZERO', 'LOW', 'NORMAL', 'HIGH'))
    bn.add_node('FIO2', 'discrete', 2, ('LOW', 'NORMAL'))
    bn.add_node('PVSAT', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))
    bn.add_node('SAO2', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))
    bn.add_node('PAP', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))
    bn.add_node('PULMEMBOLUS', 'discrete', 2, ('TRUE', 'FALSE'))
    bn.add_node('SHUNT', 'discrete', 2, ('NORMAL', 'HIGH'))
    bn.add_node('INTUBATION', 'discrete', 3, ('NORMAL', 'ESOPHAGEAL', 'ONESIDED'))
    bn.add_node('PRESS', 'discrete', 4, ('ZERO', 'LOW', 'NORMAL', 'HIGH'))
    bn.add_node('DISCONNECT', 'discrete', 2, ('TRUE', 'FALSE'))
    bn.add_node('MINVOLSET', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))
    bn.add_node('VENTMACH', 'discrete', 4, ('ZERO', 'LOW', 'NORMAL', 'HIGH'))
    bn.add_node('VENTTUBE', 'discrete', 4, ('ZERO', 'LOW', 'NORMAL', 'HIGH'))
    bn.add_node('VENTLUNG', 'discrete', 4, ('ZERO', 'LOW', 'NORMAL', 'HIGH'))
    bn.add_node('VENTALV', 'discrete', 4, ('ZERO', 'LOW', 'NORMAL', 'HIGH'))
    bn.add_node('ARTCO2', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))
    bn.add_node('CATECHOL', 'discrete', 2, ('NORMAL', 'HIGH'))
    bn.add_node('HR', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))
    bn.add_node('CO', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))
    bn.add_node('BP', 'discrete', 3, ('LOW', 'NORMAL', 'HIGH'))

    bn.add_probability('HISTORY', ['LVFAILURE'], [(('TRUE'), (0.9, 0.1)),
                                                (('FALSE'), (0.01, 0.99))])

    bn.add_probability('CVP', ['LVEDVOLUME'], [(('LOW'), (0.95, 0.04, 0.01)),
                                            (('NORMAL'), (0.04, 0.95, 0.01)),
                                            (('HIGH'), (0.01, 0.29, 0.7))])
    bn.add_probability('PCWP', ['LVEDVOLUME'], [(('LOW'), (0.95, 0.04, 0.01)),
                                                (('NORMAL'), (0.04, 0.95, 0.01)),
                                                (('HIGH'), (0.01, 0.04, 0.95))])
    bn.add_probability('HYPOVOLEMIA', [None], [((None), (0.2, 0.8))])
    bn.add_probability('LVEDVOLUME', ['HYPOVOLEMIA', 'LVFAILURE'], [(('TRUE', 'TRUE'), (0.95, 0.04, 0.01)),
                                                                    (('FALSE', 'TRUE'),
                                                                    (0.98, 0.01, 0.01)),
                                                                    (('TRUE', 'FALSE'),
                                                                    (0.01, 0.09, 0.9)),
                                                                    (('FALSE', 'FALSE'), (0.05, 0.9, 0.05))])
    bn.add_probability('LVFAILURE', [None], [((None), (0.05, 0.95))])
    bn.add_probability('STROKEVOLUME', ['HYPOVOLEMIA', 'LVFAILURE'], [(('TRUE', 'TRUE'), (0.98, 0.01, 0.01)),
                                                                    (('FALSE', 'TRUE'),
                                                                    (0.95, 0.04, 0.01)),
                                                                    (('TRUE', 'FALSE'),
                                                                    (0.5, 0.49, 0.01)),
                                                                    (('FALSE', 'FALSE'), (0.05, 0.9, 0.05))])
    bn.add_probability('ERRLOWOUTPUT', [None], [((None), (0.05, 0.95))])
    bn.add_probability('HRBP', ['ERRLOWOUTPUT', 'HR'], [(('TRUE', 'LOW'), (0.98, 0.01, 0.01)),
                                                        (('FALSE', 'LOW'),
                                                        (0.4, 0.59, 0.01)),
                                                        (('TRUE', 'NORMAL'),
                                                        (0.3, 0.4, 0.3)),
                                                        (('FALSE', 'NORMAL'),
                                                        (0.98, 0.01, 0.01)),
                                                        (('TRUE', 'HIGH'),
                                                        (0.01, 0.98, 0.01)),
                                                        (('FALSE', 'HIGH'), (0.01, 0.01, 0.98))])

    bn.add_probability('HREKG', ['ERRCAUTER', 'HR'], [(('TRUE', 'LOW'), (0.33333334, 0.33333334, 0.33333334)),
                                                    (('FALSE', 'LOW'), (0.33333334,
                                                    0.33333334, 0.33333334)),
                                                    (('TRUE', 'NORMAL'), (0.33333334,
                                                    0.33333334, 0.33333334)),
                                                    (('FALSE', 'NORMAL'),
                                                    (0.98, 0.01, 0.01)),
                                                    (('TRUE', 'HIGH'),
                                                    (0.01, 0.98, 0.01)),
                                                    (('FALSE', 'HIGH'), (0.01, 0.01, 0.98))])
    bn.add_probability('ERRCAUTER', [None], [((None), (0.1, 0.9))])
    bn.add_probability('HRSAT', ['ERRCAUTER', 'HR'], [(('TRUE', 'LOW'), (0.33333334, 0.33333334, 0.33333334)),
                                                    (('FALSE', 'LOW'), (0.33333334,
                                                    0.33333334, 0.33333334)),
                                                    (('TRUE', 'NORMAL'), (0.33333334,
                                                    0.33333334, 0.33333334)),
                                                    (('FALSE', 'NORMAL'),
                                                    (0.98, 0.01, 0.01)),
                                                    (('TRUE', 'HIGH'),
                                                    (0.01, 0.98, 0.01)),
                                                    (('FALSE', 'HIGH'), (0.01, 0.01, 0.98))])
    bn.add_probability('TPR', ['ANAPHYLAXIS'], [(('TRUE'), (0.98, 0.01, 0.01)),
                                                (('FALSE'), (0.3, 0.4, 0.3))])
    bn.add_probability('INSUFFANESTH', [None], [((None), (0.1, 0.9))])
    bn.add_probability('ANAPHYLAXIS', [None], [((None), (0.01, 0.99))])
    bn.add_probability('EXPCO2', ['ARTCO2', 'VENTLUNG'], [(('LOW', 'ZERO'), (0.97, 0.01, 0.01, 0.01)),
                                                        (('NORMAL', 'ZERO'),
                                                        (0.01, 0.97, 0.01, 0.01)),
                                                        (('HIGH', 'ZERO'),
                                                        (0.01, 0.97, 0.01, 0.01)),
                                                        (('LOW', 'LOW'),
                                                        (0.01, 0.97, 0.01, 0.01)),
                                                        (('NORMAL', 'LOW'),
                                                        (0.97, 0.01, 0.01, 0.01)),
                                                        (('HIGH', 'LOW'),
                                                        (0.01, 0.01, 0.97, 0.01)),
                                                        (('LOW', 'NORMAL'),
                                                        (0.01, 0.01, 0.97, 0.01)),
                                                        (('NORMAL', 'NORMAL'),
                                                        (0.01, 0.01, 0.97, 0.01)),
                                                        (('HIGH', 'NORMAL'),
                                                        (0.97, 0.01, 0.01, 0.01)),
                                                        (('LOW', 'HIGH'),
                                                        (0.01, 0.01, 0.01, 0.97)),
                                                        (('NORMAL', 'HIGH'),
                                                        (0.01, 0.01, 0.01, 0.97)),
                                                        (('HIGH', 'HIGH'), (0.01, 0.01, 0.01, 0.97))])
    bn.add_probability('KINKEDTUBE', [None], [((None), (0.04, 0.96))])
    bn.add_probability('MINVOL', ['INTUBATION', 'VENTLUNG'], [(('NORMAL', 'ZERO'), (0.97, 0.01, 0.01, 0.01)),
                                                            (('ESOPHAGEAL', 'ZERO'),
                                                            (0.01, 0.97, 0.01, 0.01)),
                                                            (('ONESIDED', 'ZERO'),
                                                            (0.01, 0.01, 0.97, 0.01)),
                                                            (('NORMAL', 'LOW'),
                                                            (0.01, 0.01, 0.01, 0.97)),
                                                            (('ESOPHAGEAL', 'LOW'),
                                                            (0.97, 0.01, 0.01, 0.01)),
                                                            (('ONESIDED', 'LOW'),
                                                            (0.6, 0.38, 0.01, 0.01)),
                                                            (('NORMAL', 'NORMAL'),
                                                            (0.5, 0.48, 0.01, 0.01)),
                                                            (('ESOPHAGEAL', 'NORMAL'),
                                                            (0.5, 0.48, 0.01, 0.01)),
                                                            (('ONESIDED', 'NORMAL'),
                                                            (0.97, 0.01, 0.01, 0.01)),
                                                            (('NORMAL', 'HIGH'),
                                                            (0.01, 0.97, 0.01, 0.01)),
                                                            (('ESOPHAGEAL', 'HIGH'),
                                                            (0.01, 0.01, 0.97, 0.01)),
                                                            (('ONESIDED', 'HIGH'), (0.01, 0.01, 0.01, 0.97))])
    bn.add_probability('FIO2', [None], [((None), (0.05, 0.95))])
    bn.add_probability('PVSAT', ['FIO2', 'VENTALV'], [(('LOW', 'ZERO'), (1.0, 0.0, 0.0)),
                                                    (('NORMAL', 'ZERO'),
                                                    (0.99, 0.01, 0.0)),
                                                    (('LOW', 'LOW'),
                                                    (0.95, 0.04, 0.01)),
                                                    (('NORMAL', 'LOW'),
                                                    (0.95, 0.04, 0.01)),
                                                    (('LOW', 'NORMAL'),
                                                    (1.0, 0.0, 0.0)),
                                                    (('NORMAL', 'NORMAL'),
                                                    (0.95, 0.04, 0.01)),
                                                    (('LOW', 'HIGH'),
                                                    (0.01, 0.95, 0.04)),
                                                    (('NORMAL', 'HIGH'), (0.01, 0.01, 0.98))])
    bn.add_probability('SAO2', ['PVSAT', 'SHUNT'], [(('LOW', 'NORMAL'), (0.98, 0.01, 0.01)),
                                                    (('NORMAL', 'NORMAL'),
                                                    (0.01, 0.98, 0.01)),
                                                    (('HIGH', 'NORMAL'),
                                                    (0.01, 0.01, 0.98)),
                                                    (('LOW', 'HIGH'),
                                                    (0.98, 0.01, 0.01)),
                                                    (('NORMAL', 'HIGH'),
                                                    (0.98, 0.01, 0.01)),
                                                    (('HIGH', 'HIGH'), (0.69, 0.3, 0.01))])
    bn.add_probability('PAP', ['PULMEMBOLUS'], [(('TRUE'), (0.01, 0.19, 0.8)),
                                                (('FALSE'), (0.05, 0.9, 0.05))])
    bn.add_probability('PULMEMBOLUS', [None], [((None), (0.01, 0.99))])
    bn.add_probability('SHUNT', ['INTUBATION', 'PULMEMBOLUS'], [(('NORMAL', 'TRUE'), (0.1, 0.9)),
                                                                (('ESOPHAGEAL', 'TRUE'),
                                                                (0.1, 0.9)),
                                                                (('ONESIDED', 'TRUE'),
                                                                (0.01, 0.99)),
                                                                (('NORMAL', 'FALSE'),
                                                                (0.95, 0.05)),
                                                                (('ESOPHAGEAL', 'FALSE'),
                                                                (0.95, 0.05)),
                                                                (('ONESIDED', 'FALSE'), (0.05, 0.95))])
    bn.add_probability('INTUBATION', [None], [((None), (0.92, 0.03, 0.05))])
    bn.add_probability('PRESS', ['INTUBATION', 'KINKEDTUBE', 'VENTTUBE'], [(('NORMAL', 'TRUE', 'ZERO'), (0.97, 0.01, 0.01,
                                                                                                        0.01)),
                                                                        (('ESOPHAGEAL', 'TRUE', 'ZERO'), (0.01, 0.3,
                                                                                                            0.49, 0.2)),
                                                                        (('ONESIDED', 'TRUE', 'ZERO'), (0.01, 0.01, 0.08,
                                                                                                        0.9)),
                                                                        (('NORMAL', 'FALSE', 'ZERO'), (0.01, 0.01, 0.01,
                                                                                                        0.97)),
                                                                        (('ESOPHAGEAL', 'FALSE', 'ZERO'), (0.97, 0.01,
                                                                                                            0.01, 0.01)),
                                                                        (('ONESIDED', 'FALSE', 'ZERO'), (0.1, 0.84, 0.05,
                                                                                                            0.01)),
                                                                        (('NORMAL', 'TRUE', 'LOW'), (0.05, 0.25, 0.25,
                                                                                                        0.45)),
                                                                        (('ESOPHAGEAL', 'TRUE', 'LOW'), (0.01, 0.15,
                                                                                                            0.25, 0.59)),
                                                                        (('ONESIDED', 'TRUE', 'LOW'), (0.97, 0.01, 0.01,
                                                                                                        0.01)),
                                                                        (('NORMAL', 'FALSE', 'LOW'), (0.01, 0.29, 0.3,
                                                                                                        0.4)),
                                                                        (('ESOPHAGEAL', 'FALSE', 'LOW'), (0.01, 0.01,
                                                                                                            0.08, 0.9)),
                                                                        (('ONESIDED', 'FALSE', 'LOW'), (0.01, 0.01, 0.01,
                                                                                                        0.97)),
                                                                        (('NORMAL', 'TRUE', 'NORMAL'), (0.97, 0.01, 0.01,
                                                                                                        0.01)),
                                                                        (('ESOPHAGEAL', 'TRUE', 'NORMAL'), (0.01, 0.97,
                                                                                                            0.01, 0.01)),
                                                                        (('ONESIDED', 'TRUE', 'NORMAL'), (0.01, 0.01,
                                                                                                            0.97, 0.01)),
                                                                        (('NORMAL', 'FALSE', 'NORMAL'), (0.01, 0.01,
                                                                                                            0.01, 0.97)),
                                                                        (('ESOPHAGEAL', 'FALSE', 'NORMAL'), (0.97, 0.01,
                                                                                                                0.01,
                                                                                                                0.01)),
                                                                        (('ONESIDED', 'FALSE', 'NORMAL'), (0.4, 0.58,
                                                                                                            0.01, 0.01)),
                                                                        (('NORMAL', 'TRUE', 'HIGH'), (0.2, 0.75, 0.04,
                                                                                                        0.01)),
                                                                        (('ESOPHAGEAL', 'TRUE', 'HIGH'), (0.2, 0.7, 0.09,
                                                                                                            0.01)),
                                                                        (('ONESIDED', 'TRUE', 'HIGH'), (0.97, 0.01, 0.01,
                                                                                                        0.01)),
                                                                        (('NORMAL', 'FALSE', 'HIGH'), (0.010000001,
                                                                                                        0.90000004,
                                                                                                        0.080000006,
                                                                                                        0.010000001)),
                                                                        (('ESOPHAGEAL', 'FALSE', 'HIGH'), (0.01, 0.01,
                                                                                                            0.38, 0.6)),
                                                                        (('ONESIDED', 'FALSE', 'HIGH'), (0.01, 0.01,
                                                                                                            0.01, 0.97))])
    bn.add_probability('DISCONNECT', [None], [((None), (0.1, 0.9))])
    bn.add_probability('MINVOLSET', [None], [((None), (0.05, 0.9, 0.05))])
    bn.add_probability('VENTMACH', ['MINVOLSET'], [(('LOW'), (0.05, 0.93, 0.01, 0.01)),
                                                (('NORMAL'), (0.05, 0.01, 0.93, 0.01)),
                                                (('HIGH'), (0.05, 0.01, 0.01, 0.93))])
    bn.add_probability('VENTTUBE', ['DISCONNECT', 'VENTMACH'], [(('TRUE', 'ZERO'), (0.97, 0.01, 0.01, 0.01)),
                                                                (('FALSE', 'ZERO'),
                                                                (0.97, 0.01, 0.01, 0.01)),
                                                                (('TRUE', 'LOW'),
                                                                (0.97, 0.01, 0.01, 0.01)),
                                                                (('FALSE', 'LOW'),
                                                                (0.97, 0.01, 0.01, 0.01)),
                                                                (('TRUE', 'NORMAL'),
                                                                (0.97, 0.01, 0.01, 0.01)),
                                                                (('FALSE', 'NORMAL'),
                                                                (0.01, 0.97, 0.01, 0.01)),
                                                                (('TRUE', 'HIGH'),
                                                                (0.01, 0.01, 0.97, 0.01)),
                                                                (('FALSE', 'HIGH'), (0.01, 0.01, 0.01, 0.97))])
    bn.add_probability('VENTLUNG', ['INTUBATION', 'KINKEDTUBE', 'VENTTUBE'], [(('NORMAL', 'TRUE', 'ZERO'), (0.97, 0.01,
                                                                                                            0.01, 0.01)),
                                                                            (('ESOPHAGEAL', 'TRUE', 'ZERO'), (0.95000005,
                                                                                                                0.030000001,
                                                                                                                0.010000001,
                                                                                                                0.010000001)
                                                                            ),
                                                                            (('ONESIDED', 'TRUE', 'ZERO'), (0.4, 0.58,
                                                                                                            0.01, 0.01)),
                                                                            (('NORMAL', 'FALSE', 'ZERO'), (0.3, 0.68,
                                                                                                            0.01, 0.01)),
                                                                            (('ESOPHAGEAL', 'FALSE', 'ZERO'), (0.97, 0.01,
                                                                                                                0.01, 0.01)
                                                                            ),
                                                                            (('ONESIDED', 'FALSE', 'ZERO'), (0.97, 0.01,
                                                                                                            0.01, 0.01)),
                                                                            (('NORMAL', 'TRUE', 'LOW'), (0.97, 0.01, 0.01,
                                                                                                        0.01)),
                                                                            (('ESOPHAGEAL', 'TRUE', 'LOW'), (0.97, 0.01,
                                                                                                            0.01, 0.01)),
                                                                            (('ONESIDED', 'TRUE', 'LOW'), (0.97, 0.01,
                                                                                                            0.01, 0.01)),
                                                                            (('NORMAL', 'FALSE', 'LOW'), (0.95000005,
                                                                                                            0.030000001,
                                                                                                            0.010000001,
                                                                                                            0.010000001)),
                                                                            (('ESOPHAGEAL', 'FALSE', 'LOW'), (0.5, 0.48,
                                                                                                                0.01, 0.01)
                                                                            ),
                                                                            (('ONESIDED', 'FALSE', 'LOW'), (0.3, 0.68,
                                                                                                            0.01, 0.01)),
                                                                            (('NORMAL', 'TRUE', 'NORMAL'), (0.97, 0.01,
                                                                                                            0.01, 0.01)),
                                                                            (('ESOPHAGEAL', 'TRUE', 'NORMAL'), (0.01,
                                                                                                                0.97,
                                                                                                                0.01,
                                                                                                                0.01)),
                                                                            (('ONESIDED', 'TRUE', 'NORMAL'), (0.01, 0.01,
                                                                                                                0.97, 0.01)
                                                                            ),
                                                                            (('NORMAL', 'FALSE', 'NORMAL'), (0.01, 0.01,
                                                                                                            0.01, 0.97)),
                                                                            (('ESOPHAGEAL', 'FALSE', 'NORMAL'), (0.97,
                                                                                                                0.01,
                                                                                                                0.01,
                                                                                                                0.01)),
                                                                            (('ONESIDED', 'FALSE', 'NORMAL'), (0.97, 0.01,
                                                                                                                0.01, 0.01)
                                                                            ),
                                                                            (('NORMAL', 'TRUE', 'HIGH'), (0.97, 0.01,
                                                                                                            0.01, 0.01)),
                                                                            (('ESOPHAGEAL', 'TRUE', 'HIGH'), (0.97, 0.01,
                                                                                                                0.01, 0.01)
                                                                            ),
                                                                            (('ONESIDED', 'TRUE', 'HIGH'), (0.97, 0.01,
                                                                                                            0.01, 0.01)),
                                                                            (('NORMAL', 'FALSE', 'HIGH'), (0.01, 0.97,
                                                                                                            0.01, 0.01)),
                                                                            (('ESOPHAGEAL', 'FALSE', 'HIGH'), (0.01, 0.01,
                                                                                                                0.97, 0.01)
                                                                            ),
                                                                            (('ONESIDED', 'FALSE', 'HIGH'), (0.01, 0.01,
                                                                                                            0.01, 0.97))
                                                                            ])
    bn.add_probability('VENTALV', ['INTUBATION', 'VENTLUNG'], [(('NORMAL', 'ZERO'), (0.97, 0.01, 0.01, 0.01)),
                                                            (('ESOPHAGEAL', 'ZERO'),
                                                                (0.01, 0.97, 0.01, 0.01)),
                                                            (('ONESIDED', 'ZERO'),
                                                                (0.01, 0.01, 0.97, 0.01)),
                                                            (('NORMAL', 'LOW'),
                                                                (0.01, 0.01, 0.01, 0.97)),
                                                            (('ESOPHAGEAL', 'LOW'),
                                                                (0.97, 0.01, 0.01, 0.01)),
                                                            (('ONESIDED', 'LOW'),
                                                                (0.01, 0.97, 0.01, 0.01)),
                                                            (('NORMAL', 'NORMAL'),
                                                                (0.01, 0.01, 0.97, 0.01)),
                                                            (('ESOPHAGEAL', 'NORMAL'),
                                                                (0.01, 0.01, 0.01, 0.97)),
                                                            (('ONESIDED', 'NORMAL'),
                                                                (0.97, 0.01, 0.01, 0.01)),
                                                            (('NORMAL', 'HIGH'), (0.030000001, 0.95000005, 0.010000001,
                                                                                    0.010000001)),
                                                            (('ESOPHAGEAL', 'HIGH'),
                                                                (0.01, 0.94, 0.04, 0.01)),
                                                            (('ONESIDED', 'HIGH'), (0.01, 0.88, 0.1, 0.01))])
    bn.add_probability('ARTCO2', ['VENTALV'], [(('ZERO'), (0.01, 0.01, 0.98)),
                                            (('LOW'), (0.01, 0.01, 0.98)),
                                            (('NORMAL'), (0.04, 0.92, 0.04)),
                                            (('HIGH'), (0.9, 0.09, 0.01))])
    bn.add_probability('CATECHOL', ['ARTCO2', 'INSUFFANESTH', 'SAO2', 'TPR'],
                    [(('LOW', 'TRUE', 'LOW', 'LOW'), (0.01, 0.99)),
                        (('NORMAL', 'TRUE', 'LOW', 'LOW'), (0.01, 0.99)),
                        (('HIGH', 'TRUE', 'LOW', 'LOW'), (0.01, 0.99)),
                        (('LOW', 'FALSE', 'LOW', 'LOW'), (0.01, 0.99)),
                        (('NORMAL', 'FALSE', 'LOW', 'LOW'), (0.01, 0.99)),
                        (('HIGH', 'FALSE', 'LOW', 'LOW'), (0.01, 0.99)),
                        (('LOW', 'TRUE', 'NORMAL', 'LOW'), (0.01, 0.99)),
                        (('NORMAL', 'TRUE', 'NORMAL', 'LOW'), (0.01, 0.99)),
                        (('HIGH', 'TRUE', 'NORMAL', 'LOW'), (0.01, 0.99)),
                        (('LOW', 'FALSE', 'NORMAL', 'LOW'), (0.01, 0.99)),
                        (('NORMAL', 'FALSE', 'NORMAL', 'LOW'), (0.01, 0.99)),
                        (('HIGH', 'FALSE', 'NORMAL', 'LOW'), (0.01, 0.99)),
                        (('LOW', 'TRUE', 'HIGH', 'LOW'), (0.01, 0.99)),
                        (('NORMAL', 'TRUE', 'HIGH', 'LOW'), (0.01, 0.99)),
                        (('HIGH', 'TRUE', 'HIGH', 'LOW'), (0.01, 0.99)),
                        (('LOW', 'FALSE', 'HIGH', 'LOW'), (0.05, 0.95)),
                        (('NORMAL', 'FALSE', 'HIGH', 'LOW'), (0.05, 0.95)),
                        (('HIGH', 'FALSE', 'HIGH', 'LOW'), (0.01, 0.99)),
                        (('LOW', 'TRUE', 'LOW', 'NORMAL'), (0.01, 0.99)),
                        (('NORMAL', 'TRUE', 'LOW', 'NORMAL'), (0.01, 0.99)),
                        (('HIGH', 'TRUE', 'LOW', 'NORMAL'), (0.01, 0.99)),
                        (('LOW', 'FALSE', 'LOW', 'NORMAL'), (0.05, 0.95)),
                        (('NORMAL', 'FALSE', 'LOW', 'NORMAL'), (0.05, 0.95)),
                        (('HIGH', 'FALSE', 'LOW', 'NORMAL'), (0.01, 0.99)),
                        (('LOW', 'TRUE', 'NORMAL', 'NORMAL'), (0.05, 0.95)),
                        (('NORMAL', 'TRUE', 'NORMAL', 'NORMAL'), (0.05, 0.95)),
                        (('HIGH', 'TRUE', 'NORMAL', 'NORMAL'), (0.01, 0.99)),
                        (('LOW', 'FALSE', 'NORMAL', 'NORMAL'), (0.05, 0.95)),
                        (('NORMAL', 'FALSE', 'NORMAL', 'NORMAL'), (0.05, 0.95)),
                        (('HIGH', 'FALSE', 'NORMAL', 'NORMAL'), (0.01, 0.99)),
                        (('LOW', 'TRUE', 'HIGH', 'NORMAL'), (0.05, 0.95)),
                        (('NORMAL', 'TRUE', 'HIGH', 'NORMAL'), (0.05, 0.95)),
                        (('HIGH', 'TRUE', 'HIGH', 'NORMAL'), (0.01, 0.99)),
                        (('LOW', 'FALSE', 'HIGH', 'NORMAL'), (0.05, 0.95)),
                        (('NORMAL', 'FALSE', 'HIGH', 'NORMAL'), (0.05, 0.95)),
                        (('HIGH', 'FALSE', 'HIGH', 'NORMAL'), (0.01, 0.99)),
                        (('LOW', 'TRUE', 'LOW', 'HIGH'), (0.7, 0.3)),
                        (('NORMAL', 'TRUE', 'LOW', 'HIGH'), (0.7, 0.3)),
                        (('HIGH', 'TRUE', 'LOW', 'HIGH'), (0.1, 0.9)),
                        (('LOW', 'FALSE', 'LOW', 'HIGH'), (0.7, 0.3)),
                        (('NORMAL', 'FALSE', 'LOW', 'HIGH'), (0.7, 0.3)),
                        (('HIGH', 'FALSE', 'LOW', 'HIGH'), (0.1, 0.9)),
                        (('LOW', 'TRUE', 'NORMAL', 'HIGH'), (0.7, 0.3)),
                        (('NORMAL', 'TRUE', 'NORMAL', 'HIGH'), (0.7, 0.3)),
                        (('HIGH', 'TRUE', 'NORMAL', 'HIGH'), (0.1, 0.9)),
                        (('LOW', 'FALSE', 'NORMAL', 'HIGH'), (0.95, 0.05)),
                        (('NORMAL', 'FALSE', 'NORMAL', 'HIGH'), (0.99, 0.01)),
                        (('HIGH', 'FALSE', 'NORMAL', 'HIGH'), (0.3, 0.7)),
                        (('LOW', 'TRUE', 'HIGH', 'HIGH'), (0.95, 0.05)),
                        (('NORMAL', 'TRUE', 'HIGH', 'HIGH'), (0.99, 0.01)),
                        (('HIGH', 'TRUE', 'HIGH', 'HIGH'), (0.3, 0.7)),
                        (('LOW', 'FALSE', 'HIGH', 'HIGH'), (0.95, 0.05)),
                        (('NORMAL', 'FALSE', 'HIGH', 'HIGH'), (0.99, 0.01)),
                        (('HIGH', 'FALSE', 'HIGH', 'HIGH'), (0.3, 0.7))])
    bn.add_probability('HR', ['CATECHOL'], [(('NORMAL'), (0.05, 0.9, 0.05)),
                                            (('HIGH'), (0.01, 0.09, 0.9))])
    bn.add_probability('CO', ['HR', 'STROKEVOLUME'], [(('LOW', 'LOW'), (0.98, 0.01, 0.01)),
                                                    (('NORMAL', 'LOW'),
                                                    (0.95, 0.04, 0.01)),
                                                    (('HIGH', 'LOW'),
                                                    (0.8, 0.19, 0.01)),
                                                    (('LOW', 'NORMAL'),
                                                    (0.95, 0.04, 0.01)),
                                                    (('NORMAL', 'NORMAL'),
                                                    (0.04, 0.95, 0.01)),
                                                    (('HIGH', 'NORMAL'),
                                                    (0.01, 0.04, 0.95)),
                                                    (('LOW', 'HIGH'),
                                                    (0.3, 0.69, 0.01)),
                                                    (('NORMAL', 'HIGH'),
                                                    (0.01, 0.3, 0.69)),
                                                    (('HIGH', 'HIGH'), (0.01, 0.01, 0.98))])
    bn.add_probability('BP', ['CO', 'TPR'], [(('LOW', 'LOW'), (0.98, 0.01, 0.01)),
                                            (('NORMAL', 'LOW'), (0.98, 0.01, 0.01)),
                                            (('HIGH', 'LOW'), (0.9, 0.09, 0.01)),
                                            (('LOW', 'NORMAL'), (0.98, 0.01, 0.01)),
                                            (('NORMAL', 'NORMAL'), (0.1, 0.85, 0.05)),
                                            (('HIGH', 'NORMAL'), (0.05, 0.2, 0.75)),
                                            (('LOW', 'HIGH'), (0.3, 0.6, 0.1)),
                                            (('NORMAL', 'HIGH'), (0.05, 0.4, 0.55)),
                                            (('HIGH', 'HIGH'), (0.01, 0.09, 0.9))])
    print(bn.get_probability('BP'))


if __name__ == '__main__':
    main()
    # test()

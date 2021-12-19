from typing import Union, List, Tuple, Dict, Set, Callable
from BayesNet import BayesNet
from enum import Enum
import pandas as pd
import copy
import json
import os

class Heuristics(Enum):
    MIN_ORDER='MIN_ORDER'
    MIN_FILL='MIN_ORDER'
    RANDOM='RANDOM'

class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net) # type: ignore
        else:
            self.bn = net

    # TODO: This is where your methods should go
    
    def d_seperation(self, X: List[str], Y: List[str], Z: List[str]) -> bool:
        """
        Gets the d-seperation given the three lists of variables: X, Y and Z,
        i.e. returns true if X is independent of Y given Z in the self.bn

        :param X: variable list X
        :param Y: variable list Y
        :param Z: variable list Z

        :return: dsep(X, Y, Z)
        """

        temp_bn = copy.deepcopy(self.bn)

        return temp_bn.d_sep(set(X), set(Y), set(Z)) # type: ignore

    def ordering_min_degree(self, X: List[str]) -> List[str]:
        """
        Gets the ordering of the variable list X
        based on min-degree heuristics

        :param X: variables in network N

        :return: an ordering PI of variables X
        """
        temp_bn = copy.deepcopy(self.bn)

        return temp_bn.min_degree(X) # type: ignore

    def ordering_min_fill(self, X: List[str]) -> List[str]:
        """
        Gets the ordering of the variable list X
        based on min-degree heuristics

        :param X: variable list X

        :return: an ordering PI of variables X
        """
        temp_bn = copy.deepcopy(self.bn)

        return temp_bn.min_fill(X) # type: ignore

    def network_prune(self, Q: List[str], e: List[tuple[str, bool]]) -> None:
        """
        Node- and edgeprunes the network self.bn
        Based on query variables Q and evidence e

        :param Q: query variables
        :param e: evidence
        """
        temp_bn = copy.deepcopy(self.bn)

        return temp_bn.net_prune(set(Q), e) # type: ignore

    def marginal_dist(self, Q: List[str], e: List[tuple[str, bool] or None], pi: List[str]) -> pd.DataFrame:
        """
        Computes the marginal distribution P(Q|e)
        given the query variables Q and possibly empty evidence e

        :param Q: query variables
        :param pi: ordering of network variables not in Q
        :param e: evidence

        :return: TODO
        """
        temp_bn = copy.deepcopy(self.bn)

        return temp_bn.marginal_distrib(Q, e, pi) # type: ignore

    def map_and_mpe(self, order_function: Heuristics, e: List[tuple[str, bool]], M: List[str]=[]): # TODO: add types
        """
        Computes the most likely instantiations of M
        given possibly empty set of query variables M and an evidence E

        :param order_function: function for ordering parameters
        :param M: query variables
        :param E: evidence

        :return: TODO
        """
        temp_bn = copy.deepcopy(self.bn)

        if Heuristics.MIN_FILL == order_function: order=temp_bn.min_fill
        elif Heuristics.MIN_ORDER == order_function: order=temp_bn.min_degree
        else: order=temp_bn.random_order

        return temp_bn.map_and_mpe(order, e, M)

    # def create_random_bn(self) -> None:
    #     bn = BayesNet()
    #     return bn.generate_random(10)
def task_2() -> None:
    directory = "./test_data/task_2"

    for filename in os.listdir(directory):
        if filename.endswith("101.xml"): 
            # print(os.path.join(directory, filename))
            file = os.path.join(directory, filename)
            bn =  BNReasoner(file)
            print(file)
            print(bn.map_and_mpe(order_function=Heuristics.MIN_ORDER, e=[('node1', False), ('node10', True)]))
            continue
        else:
            continue
            

if __name__ =='__main__':
    # bn = BayesNet()
    # bn.generate_random(10)
    # bn.draw_structure()
    task_2()

    lecture_Example = BNReasoner("testing/lecture_Example.BIFXML")
    lecture_Example_2 = BNReasoner("testing/lecture_Example2.BIFXML")
    dog_problem = BNReasoner("testing/dog_problem.BIFXML")
    print(dog_problem.d_seperation(['family-out'], ['hear-bark'], ['dog-out']))
    # print(dog_problem.ordering_min_degree())
    # print(dog_problem.ordering_min_fill())
    dog_problem.network_prune(['family-out'],[('dog-out', True), ('hear-bark', False)])
    print(lecture_Example.marginal_dist(['Slippery Road?',  'Wet Grass?'], [('Winter?', True), ('Sprinkler?', False)], ['Winter?', 'Rain?', 'Sprinkler?']))
    print(lecture_Example_2.map_and_mpe(order_function=Heuristics.MIN_ORDER, e=[('O', True)], M=['I',  'J']))
    print(lecture_Example_2.map_and_mpe(order_function=Heuristics.MIN_ORDER, e=[('O', False), ('J', True)]))

    # create graphs
    # bn = BayesNet()
    # bn.load_from_bifxml("testing\\dog_problem.BIFXML")
    # interaction_graph = bn.get_interaction_graph()
    # bn.draw_structure()
    # bn.draw_graph(interaction_graph)
    # print(nx.d_separated(bn.structure, {'family-out'}, {'hear-bark'}, {'dog-out'}))

    bn = BayesNet()
    bn.load_from_bifxml("testing/diseases.BIFXML")
    bn.draw_structure()
    # disease_problem = BNReasoner('testing/diseases.BIFXML')
    
    # TESTING -------------------------

    test_problems = ['dog_problem', 'lecture_example', 'lecture_example2']
    # test_problems = ['lecture_example2']

    # delta value for accuracy in floating point outputs, i.e. the output will have to be within +/- one delta of the desired result
    DELTA = 0.01

    for prob in test_problems:
        # create a reasoner object based on the problem BIFXML file
        reasoner = BNReasoner('testing\\' + prob + '.BIFXML')

        # get test data from json
        data = {}
        with open('test_data\\' + prob + '.json', 'r') as infile:
            data = json.load(infile)

        # test d-seperation
        for X, Y, Z, result in data['d_sep']:
            assert reasoner.d_seperation(X, Y, Z) == result

        # test min-degree ordering
        for X, result in data['min_degree']:
            assert reasoner.ordering_min_degree(X) == result

        # test min-fill ordering
        for X, result in data['min_fill']:
            assert reasoner.ordering_min_fill(X) == result

        # test network pruning
        for Q, e, vars, edges in data['net_prune']:
            temp_bn = copy.deepcopy(reasoner.bn)
            assert isinstance(temp_bn, BayesNet)

            e_list: List[Tuple] = list(tuple(x) for x in e)
            edge_list: List[Tuple] = list(tuple(x) for x in edges)
            
            temp_bn.net_prune(set(Q), e)
            # verify whether the pruned network has the same variables as the computed test data
            assert temp_bn.get_all_variables() == vars
            assert temp_bn.get_all_edges() == edge_list

        # test marginal distributions
        for Q, e, pi, result in data['marginal_distrib']:
            e_list: List[Tuple] = list(tuple(x) for x in e)

            # the output CPT from the marginal distribution
            df = reasoner.marginal_dist(Q, e_list, pi)  # type: ignore

            # the last row, which is where all variables are true
            output = df.iloc[-1].loc['p']

            result_float = float(result[0])

            # test whether the probability value of the last row is within a delta margin of the test value
            assert (result_float - DELTA) < output < (result_float + DELTA)

        # test MAP and MPE
        for _ in data['map_and_mpe']:
            pass

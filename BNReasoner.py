from typing import Union, List, Tuple, Dict, Set
from BayesNet import BayesNet
import pandas as pd
import networkx as nx
import copy
import json

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

    def map_and_mpe(self, M: List[str] or None, E: List[tuple[str, bool]]): # TODO: add types
        """
        Computes the most likely instantiations of M
        given possibly empty set of query variables M and an evidence E

        :param M: query variables
        :param E: evidence

        :return: TODO
        """
        temp_bn = copy.deepcopy(self.bn)

        return temp_bn.map_and_mpe(M, E)


if __name__ =='__main__':
    lecture_Example = BNReasoner("testing/lecture_Example.BIFXML")
    lecture_Example_2 = BNReasoner("testing/lecture_Example2.BIFXML")
    dog_problem = BNReasoner("testing/dog_problem.BIFXML")
    print(dog_problem.d_seperation(['family-out'], ['hear-bark'], ['dog-out']))
    # print(dog_problem.ordering_min_degree())
    # print(dog_problem.ordering_min_fill())
    dog_problem.network_prune(['family-out'],[('dog-out', True), ('hear-bark', False)])
    print(lecture_Example.marginal_dist(['Slippery Road?',  'Wet Grass?'], [('Winter?', True), ('Sprinkler?', False)], ['Winter?', 'Rain?', 'Sprinkler?']))
    print(lecture_Example_2.map_and_mpe(M=['I',  'J'], E=[('O', True)]))

    # create graphs
    # bn = BayesNet()
    # bn.load_from_bifxml("testing\\dog_problem.BIFXML")
    # interaction_graph = bn.get_interaction_graph()
    # bn.draw_structure()
    # bn.draw_graph(interaction_graph)
    # print(nx.d_separated(bn.structure, {'family-out'}, {'hear-bark'}, {'dog-out'}))

    # TESTING -------------------------

    # test_problems = ['dog_problem', 'lecture_example', 'lecture_example2']
    test_problems = ['dog_problem']

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
        for X, output in data['min_degree']:
            assert reasoner.ordering_min_degree(X) == output

        # test min-fill ordering
        for X, output in data['min_fill']:
            try:
                assert reasoner.ordering_min_fill(X) == output
            except AssertionError:
                print(f'ERROR: {X} -> {output}')
        
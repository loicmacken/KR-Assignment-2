from typing import Union, List, Tuple, Dict, Set
from BayesNet import BayesNet
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
            self.bn.load_from_bifxml(net)
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

        return temp_bn.d_sep(set(X), set(Y), set(Z))    # type: ignore

    def ordering_min_degree(self) -> List[str]:
        """
        Gets the ordering of the variable list X
        based on min-degree heuristics

        :param X: variables in network N

        :return: an ordering PI of variables X
        """
        temp_bn = copy.deepcopy(self.bn)

        return temp_bn.min_degree()                    # type: ignore

    def ordering_min_fill(self) -> List[str]:
        """
        Gets the ordering of the variable list X
        based on min-degree heuristics

        :param X: variable list X

        :return: an ordering PI of variables X
        """
        temp_bn = copy.deepcopy(self.bn)

        return temp_bn.min_fill()                      # type: ignore

    def network_prune(self, Q: List[str], e: List[tuple[str, bool]]) -> None:
        """
        Node- and edgeprunes the network self.bn
        Based on query variables Q and evidence e

        :param Q: query variables
        :param e: evidence
        """
        temp_bn = copy.deepcopy(self.bn)

        temp_bn.net_prune(set(Q), e)                    # type: ignore

    def marginal_dist(self, Q: List[str], pi: List[str], e: List[tuple[str, bool]]): # TODO: add types
        """
        Computes the marginal distribution P(Q|e)
        given the query variables Q and possibly empty evidence e

        :param Q: query variables
        :param pi: ordering of network variables not in Q
        :param e: evidence

        :return: TODO
        """
        temp_bn = copy.deepcopy(self.bn)
        cpt = temp_bn.get_cpt(Q[0])
        temp_bn.sum_out(cpt, Q[1])                     # type: ignore

        temp_bn.marginal_dist(Q, pi, e)                 # type: ignore

    def map_and_mpe(self, Q: List[str], R: List[tuple[str, bool]]): # TODO: add types
        """
        Computes the most likely instantiations of Q
        given possibly empty set of query variables Q and an evidence E

        :param Q: query variables
        :param E: evidence

        :return: TODO
        """
        raise ValueError


if __name__ =='__main__':
    # get the names of all test problems
    test_problems = ['dog_problem', 'lecture_example', 'lecture_example2']

    for prob in test_problems:
        # create a reasoner object based on the problem BIFXML file
        reasoner = BNReasoner('testing\\' + prob + '.BIFXML')

        # get test data from json
        data = {}
        with open('test_data\\' + prob + '.json', 'r') as infile:
            data = json.load(infile)

        # test d-seperation
        for X, Y, Z, result in data['d_sep']:
            print(f'Testing d-sep of {prob}:\n X, Y, Z = {X}, {Y}, {Z}\n result = {result}')
            assert reasoner.d_seperation(X, Y, Z) == result
        
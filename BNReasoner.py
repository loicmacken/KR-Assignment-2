from typing import Union, List
from BayesNet import BayesNet
from enum import Enum
import helper_functions as helpers
import pandas as pd
import copy


class Heuristics(Enum):
    MIN_ORDER='MIN_ORDER'
    MIN_FILL='MIN_FILL'
    RANDOM='RANDOM'

class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if isinstance(net, str):
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    def draw_structure(self) -> None:
        temp_bn = copy.deepcopy(self.bn)
        assert isinstance(temp_bn, BayesNet)

        temp_bn.draw_structure()

    def draw_interaction_graph(self) -> None:
        temp_bn = copy.deepcopy(self.bn)
        assert isinstance(temp_bn, BayesNet)

        interaction_graph = temp_bn.draw_structure()
        assert isinstance(interaction_graph, BayesNet)

        BayesNet.draw_graph(interaction_graph)

    def d_seperation(self, X: List[str], Y: List[str], Z: List[str]) -> bool:
        """
        Gets the d-seperation given the three lists of variables: X, Y and Z,
        i.e. returns true if X is independent of Y given Z in the self.bn.

        :param X: variable list X
        :param Y: variable list Y
        :param Z: variable list Z

        :return: dsep(X, Y, Z)
        """

        temp_bn = copy.deepcopy(self.bn)
        assert isinstance(temp_bn, BayesNet)

        return temp_bn.d_sep(set(X), set(Y), set(Z))

    def ordering_min_degree(self, X: List[str]) -> List[str]:
        """
        Gets the ordering of the variable list X
        based on min-degree heuristics.

        :param X: variables in network N

        :return: an ordering PI of variables X
        """
        temp_bn = copy.deepcopy(self.bn)
        assert isinstance(self.bn, BayesNet)

        return temp_bn.min_degree(X)

    def ordering_min_fill(self, X: List[str]) -> List[str]:
        """
        Gets the ordering of the variable list X
        based on min-fill heuristics.

        :param X: variable list X

        :return: an ordering PI of variables X
        """
        temp_bn = copy.deepcopy(self.bn)
        assert isinstance(temp_bn, BayesNet)

        return temp_bn.min_fill(X)

    def network_prune(self, Q: List[str], e: List[tuple[str, bool]]) -> None:
        """
        Node- and edgeprunes the network self.bn
        Based on query variables Q and evidence e

        :param Q: set of query variables
        :param e: list of evidence
        """
        temp_bn = copy.deepcopy(self.bn)
        assert isinstance(temp_bn, BayesNet)

        temp_bn.net_prune(set(Q), e)

    def marginal_dist(self, Q: List[str], e: List[tuple[str, bool] or None], order_function: Heuristics) -> pd.DataFrame:
        """
        Computes the marginal distribution P(Q|e)
        given the query variables Q and possibly empty evidence e.
        If e is empty it returns prior marginals,
        else it returns posterior marginals given e.

        :param Q: query variables
        :param pi: ordering of network variables not in Q
        :param e: evidence

        :return: the prior or posterior marginals
        """
        temp_bn = copy.deepcopy(self.bn)
        assert isinstance(self.bn, BayesNet)

        pi = temp_bn.get_all_variables()

        for q in Q: 
            if q in pi: 
                pi.remove(q)

        if Heuristics.MIN_FILL == order_function: order_pi=temp_bn.min_fill(pi)
        elif Heuristics.MIN_ORDER == order_function: order_pi=temp_bn.min_degree(pi)
        else: order_pi=temp_bn.random_order(pi)

        return temp_bn.marginal_distrib(Q, e, order_pi)

    def map_and_mpe(self, order_function: Heuristics, e: List[tuple[str, bool]], M: List[str]=[]) -> pd.DataFrame:
        """
        Computes the most likely instantiations given evidence and an ordering function (heuristic).
        If M is empty, returns MPE, else it returns MAP

        :param order_function: function for ordering parameters (heuristic)
        :param M: variables in network that we do not want to eliminate or None in case of MPE
        :param e: evidence

        :return: the marginal distribution of MAP_p(M, E) or MPE_p(E)
        """
        temp_bn = copy.deepcopy(self.bn)
        assert isinstance(self.bn, BayesNet)

        if Heuristics.MIN_FILL == order_function: order=temp_bn.min_fill
        elif Heuristics.MIN_ORDER == order_function: order=temp_bn.min_degree
        else: order=temp_bn.random_order

        return temp_bn.map_and_mpe(order, e, M)

if __name__ =='__main__':
    pass
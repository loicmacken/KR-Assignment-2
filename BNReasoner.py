from typing import Union
from BayesNet import BayesNet


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

    def d_sep(self, X: list[str], Y: list[str], Z: list[str]) -> bool:
        """
        Gets the d-seperation given the three lists of variables: X, Y and Z,
        i.e. returns true if X is independent of Y given Z in the self.bn

        :param X: variable list X
        :param Y: variable list Y
        :param Z: variable list Z

        :return: dsep(X, Y, Z)
        """
        raise ValueError

    def ordering_min_degree(self, X: list[str]) -> list[str]:
        """
        Gets the ordering of the variable list X
        based on min-degree heuristics

        :param X: variable list X

        :return: an ordering PI of variables X
        """
        raise ValueError

    def ordering_min_fill(self, X: list[str]) -> list[str]:
        """
        Gets the ordering of the variable list X
        based on min-degree heuristics

        :param X: variable list X

        :return: an ordering PI of variables X
        """
        raise ValueError

    def network_prune(self, Q: list[str], e: list[tuple[str, bool]]) -> bool:
        """
        Node- and edgeprunes the network self.bn
        Based on query variables Q and evidence e

        :param Q: query variables
        :param e: evidence

        :return: is successful
        """
        raise ValueError

    def marginal_dist(self, Q: list[str], e: list[tuple[str, bool]]): # TODO: add types
        """
        Computes the marginal distribution P(Q|e)
        given the query variables Q and possibly empty evidence e

        :param Q: query variables
        :param e: evidence

        :return: TODO
        """
        raise ValueError

    def map_and_mep(self, Q: list[str], e: list[tuple[str, bool]]): # TODO: add types
        """
        Computes the most likely instantiations of Q
        given possibly empty set of query variables Q and an evidence e

        :param Q: query variables
        :param e: evidence

        :return: TODO
        """
        raise ValueError


if __name__ =='__main__':
    pass

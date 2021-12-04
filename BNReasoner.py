from typing import Union
from BayesNet import BayesNet
import copy


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

    @staticmethod
    def get_leave_nodes(net: BayesNet) -> list[str]:
        vars: set[str] = set(net.get_all_variables())

        for node, _ in net.structure.edges:
            vars.remove(node)
        
        return list(vars)

    @staticmethod
    def get_root_nodes(net: BayesNet) -> list[str]:
        vars: set[str] = set(net.get_all_variables())

        for _, node in net.structure.edges:
            vars.remove(node)

        return list(vars)

    @staticmethod
    def get_node_parents(net: BayesNet, W: str) -> list[str]:
        parents: list[str] = []
        for parent, child in net.structure.edges:
            if child == W:
                parents.append(parent)

        return parents

    @staticmethod
    def get_paths(X: list[str], Y: list[str]) -> list[list[tuple[str, str]]]:
        pass

    @staticmethod
    def get_valve_type(net: BayesNet, W: str) -> str:
        parents: list[str] = BNReasoner.get_node_parents(net, W)
        if len(parents) > 1:
            return 'con'
        
        children: list[str] = net.get_children(W)
        if len(children) > 1:
            return 'div'

        return 'seq'

    def d_sep(self, X: list[str], Y: list[str], Z: list[str]) -> bool:
        """
        Gets the d-seperation given the three lists of variables: X, Y and Z,
        i.e. returns true if X is independent of Y given Z in the self.bn

        :param X: variable list X
        :param Y: variable list Y
        :param Z: variable list Z

        :return: dsep(X, Y, Z)
        """
        X_set = set(X)
        Y_set = set(Y)
        Z_set = set(Z)
        XYZ_set = X_set.union(Y_set, Z_set)

        temp_bn: BayesNet = copy.deepcopy(self.bn)

        temp_bn.draw_structure()

        # delete leaf nodes W not part of X U Y U Z
        while(True):
            leaves: set[str] = set(BNReasoner.get_leave_nodes(temp_bn))
            leaves_to_delete = leaves - XYZ_set

            if not leaves_to_delete: break

            for leaf in leaves_to_delete:
                temp_bn.del_var(leaf)

        temp_bn.draw_structure()

        # delete all edges outgoing from Z
        for var in Z:
            for child in temp_bn.get_children(var):
                temp_bn.del_edge((var, child))

        temp_bn.draw_structure()

        roots: list[str] = BNReasoner.get_root_nodes(temp_bn)

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

    def map_and_mpe(self, Q: list[str], e: list[tuple[str, bool]]): # TODO: add types
        """
        Computes the most likely instantiations of Q
        given possibly empty set of query variables Q and an evidence e

        :param Q: query variables
        :param e: evidence

        :return: TODO
        """
        raise ValueError


if __name__ =='__main__':
    dog_problem = BayesNet()
    dog_problem.load_from_bifxml("testing\\dog_problem.BIFXML")
    dog_reasoner = BNReasoner(dog_problem)
    dog_reasoner.d_sep(['family-out'], ['hear-bark'], ['dog-out'])
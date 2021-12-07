from typing import Union, List, Tuple, Dict, Set
from BayesNet import BayesNet
import networkx as nx
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

    # @staticmethod
    # def get_leaf_nodes(net: BayesNet) -> set[str]:
    #     vars = set(net.get_all_variables())

    #     for node, _ in net.structure.edges:
    #         if node in vars:
    #             vars.remove(node)
        
    #     return vars

    # @staticmethod
    # def get_root_nodes(net: BayesNet) -> set[str]:
    #     vars: set[str] = set(net.get_all_variables())

    #     for _, node in net.structure.edges:
    #         if node in vars:
    #             vars.remove(node)

    #     return vars

    # @staticmethod
    # def get_node_parents(net: BayesNet, W: str) -> set[str]:
    #     parents: set[str] = set()
    #     for parent, child in net.structure.edges:
    #         if child == W:
    #             parents.add(parent)

    #     return parents

    # @staticmethod
    # def get_paths(X: list[str], Y: list[str]) -> list[list[tuple[str, str]]]:
    #     pass

    # @staticmethod
    # def get_valve_type(net: BayesNet, W: str) -> str:
    #     parents: set[str] = BNReasoner.get_node_parents(net, W)
    #     if len(parents) > 1:
    #         return 'con'
        
    #     children: set[str] = set(net.get_children(W))
    #     if len(children) > 1:
    #         return 'div'

    #     return 'seq'

    # @staticmethod
    # def reachable_nodes(net: BayesNet, X: set[str], Z: set[str]) -> set[str]:
    #     """
        
    #     """
    #     # Phase I: insert all ancestors of Z into A

    #     L = Z       # nodes to be visited
    #     A = set()   # ancestors of Z
    #     while L:
    #         Y = next(iter(A))
    #         L = L - {Y}
    #         if Y not in A:
    #             L = L | BNReasoner.get_node_parents(net, Y) # Y's parents need to be visited
    #         A = A | {Y} # Y is ancestor of evidence
        
    #     # Phase II: traverse active trails starting from X
    #     # net.structure.

    # @staticmethod
    # def is_connected(net: BayesNet, X: set[str], Y: set[str]) -> bool:
    #     visited = set()
    #     for var in X:
    #         # while(True):
    #             # edges = BNReasoner.get_node_parents(net, var) | set(net.get_children(var))
    #         if BNReasoner._get_connections(net, var, Y, visited):
    #             return True
    #     return False

    # @staticmethod
    # def _get_connections(net: BayesNet, var: str, Y: set[str], visited: set[str]) -> bool:
    #     if var in Y: return True

    #     visited.add(var)
    #     edges = BNReasoner.get_node_parents(net, var) | set(net.get_children(var)) - visited

    #     for edge in edges:
    #         return BNReasoner._get_connections(net, edge, Y, visited)
    #     return False




    # def d_sep(self, X: list[str], Y: list[str], Z: list[str]) -> bool:
    #     """
    #     Gets the d-seperation given the three lists of variables: X, Y and Z,
    #     i.e. returns true if X is independent of Y given Z in the self.bn

    #     :param X: variable list X
    #     :param Y: variable list Y
    #     :param Z: variable list Z

    #     :return: dsep(X, Y, Z)
    #     """
    #     X_set = set(X)
    #     Y_set = set(Y)
    #     Z_set = set(Z)
    #     XYZ_set = X_set | Y_set | Z_set

    #     temp_bn: BayesNet = copy.deepcopy(self.bn)

    #     temp_bn.draw_structure()

    #     # delete leaf nodes W not part of X U Y U Z
    #     while(True):
    #         leaves: set[str] = set(BNReasoner.get_leaf_nodes(temp_bn))
    #         leaves_to_delete = leaves - XYZ_set

    #         if not leaves_to_delete: 
    #             break

    #         for leaf in leaves_to_delete:
    #             temp_bn.del_var(leaf)

    #     leaves: set[str] = set(BNReasoner.get_leaf_nodes(temp_bn))

    #     temp_bn.draw_structure()

    #     # delete all edges outgoing from Z
    #     for var in Z:
    #         for child in temp_bn.get_children(var):
    #             temp_bn.del_edge((var, child))

    #     temp_bn.draw_structure()

    #     # moralize
    #     for var in temp_bn.get_all_variables():
    #         parents = list(BNReasoner.get_node_parents(temp_bn, var))
    #         if len(parents) > 1:
    #             for i in range(len(parents) - 1):
    #                 temp_bn.add_edge((parents[i], parents[i+1]))

    #     temp_bn.draw_structure()

    #     # remove givens
    #     for var in Z:
    #         for parent in BNReasoner.get_node_parents(temp_bn, var):
    #             temp_bn.del_edge((parent, var))
    #         temp_bn.del_var(var)

    #     temp_bn.draw_structure()

    #     return not BNReasoner.is_connected(temp_bn, X_set, Y_set)

    #     # nx.path_graph()

    #     # roots: set[str] = BNReasoner.get_root_nodes(temp_bn)

    #     # raise ValueError

    
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

    def ordering_min_degree(self, X: List[str]) -> List[str]:
        """
        Gets the ordering of the variable list X
        based on min-degree heuristics

        :param X: variable list X

        :return: an ordering PI of variables X
        """
        temp_bn = copy.deepcopy(self.bn)

        return temp_bn.min_degree(X)                    # type: ignore

    def ordering_min_fill(self, X: List[str]) -> List[str]:
        """
        Gets the ordering of the variable list X
        based on min-degree heuristics

        :param X: variable list X

        :return: an ordering PI of variables X
        """
        temp_bn = copy.deepcopy(self.bn)

        return temp_bn.min_fill(X)                      # type: ignore

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
    dog_problem = BNReasoner("testing\\dog_problem.BIFXML")
    print(dog_problem.d_seperation(['family-out'], ['hear-bark'], ['dog-out']))
    print(dog_problem.ordering_min_degree(['family-out', 'hear-bark', 'dog-out']))
    print(dog_problem.ordering_min_fill(['family-out', 'hear-bark', 'dog-out']))
    dog_problem.network_prune(['family-out', 'hear-bark', 'dog-out'],[('light-out', True)])
    dog_problem.marginal_dist([], [], [])


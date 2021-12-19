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

    def map_and_mpe(self, order_function: Heuristics, e: List[tuple[str, bool]], M: List[str]=[]) -> pd.DataFrame:
        """
        Computes the most likely instantiations of M
        given possibly empty set of query variables M and an evidence E

        :param order_function: function for ordering parameters
        :param M: query variables
        :param E: evidence

        :return: Dataframe with most likely instantiations and probability
        """
        print(order_function)
        temp_bn = copy.deepcopy(self.bn)

        if Heuristics.MIN_FILL == order_function: order=temp_bn.min_fill
        elif Heuristics.MIN_ORDER == order_function: order=temp_bn.min_degree
        else: order=temp_bn.random_order

        return temp_bn.map_and_mpe(order, e, M)

# def task_2() -> None:
#     directory = "./test_data/task_2"

#     for filename in os.listdir(directory):
#         if filename.endswith("101.xml"): 
#             # print(os.path.join(directory, filename))
#             file = os.path.join(directory, filename)
#             bn =  BNReasoner(file)
#             print(file)
#             print(bn.map_and_mpe(order_function=Heuristics.MIN_ORDER, e=[('node1', False), ('node10', True)]))
#             continue
#         else:
#             continue

def task_2() -> None:
    start_num_nodes = 10
    start_num_roots = 2
    start_num_min_edges = 2
    start_num_max_edges = 4
    increase_nodes = 10
    num_tries = 5

    # change variables
    for run_number in range(num_tries):
        bn = BayesNet()
        result = bn.generate_random(number=start_num_nodes, 
            n_roots=start_num_roots, max_edges=start_num_max_edges, min_edges=start_num_min_edges)
        bn.draw_structure_sample()
        print(bn.get_all_cpts())
        reasoner = BNReasoner(bn)

        start_num_nodes+=increase_nodes
        for method in Heuristics:
            print(method)
            reasoner.map_and_mpe(order_function=method, e=[('node1', False), ('node2', False)], M=['node3', 'node4'])
            helpers.save_data(method.value, run_number, result)

    # bn = BayesNet()
    # bn.generate_random(5, n_roots=2)
    # bn.draw_structure_sample()
    # int_grah = bn.get_interaction_graph()
    # bn.draw_graph(int_grah)
    # print(bn.get_all_cpts())
    # bn_2 = BayesNet()
    # bn_2.generate_random()
    # bn_2.draw_structure_sample()
    # print(bn_2.get_all_cpts())
    # bn_3 = BayesNet()
    # bn_3.generate_random(20, n_roots=4)
    # bn_3.draw_structure_sample()
    # print(bn_3.get_all_cpts())

def task_3() -> None:
    # draw graph
    bn = BayesNet()
    bn.load_from_bifxml("testing/diseases.BIFXML")
    bn.draw_structure()

    # draw interaction graph
    interaction_graph = bn.get_interaction_graph()
    bn.draw_graph(interaction_graph)

    disease_problem = BNReasoner('testing/diseases.BIFXML')       

    # priori marginal query.
    # posterior marginal query.
    # MAP query
    # MEP query
    disease_problem.map_and_mpe(order_function=Heuristics.MIN_ORDER, e=[('gender', False), ('job', True)])

if __name__ =='__main__':

    task_2()

    # lecture_Example = BNReasoner("testing/lecture_Example.BIFXML")
    # lecture_Example_2 = BNReasoner("testing/lecture_Example2.BIFXML")
    # dog_problem = BNReasoner("testing/dog_problem.BIFXML")
    # print(dog_problem.d_seperation(['family-out'], ['hear-bark'], ['dog-out']))
    # # print(dog_problem.ordering_min_degree())
    # # print(dog_problem.ordering_min_fill())
    # dog_problem.network_prune(['family-out'],[('dog-out', True), ('hear-bark', False)])
    # print(lecture_Example.marginal_dist(['Slippery Road?',  'Wet Grass?'], [('Winter?', True), ('Sprinkler?', False)], ['Winter?', 'Rain?', 'Sprinkler?']))
    # print(lecture_Example_2.map_and_mpe(order_function=Heuristics.MIN_ORDER, e=[('O', True)], M=['I',  'J']))

    # print(lecture_Example_2.map_and_mpe(order_function=Heuristics.MIN_ORDER, e=[('O', False), ('J', False)]))
    # print(lecture_Example_2.map_and_mpe(order_function=Heuristics.MIN_FILL, e=[('O', False), ('J', False)]))
    # print(lecture_Example_2.map_and_mpe(order_function=Heuristics.RANDOM, e=[('O', False), ('J', False)]))

    # print(lecture_Example.map_and_mpe(order_function=Heuristics.MIN_ORDER, e=[('Winter?', True), ('Sprinkler?', False)]))
    # print(lecture_Example.map_and_mpe(order_function=Heuristics.MIN_FILL, e=[('Winter?', True), ('Sprinkler?', False)]))
    # print(lecture_Example.map_and_mpe(order_function=Heuristics.RANDOM, e=[('Winter?', True), ('Sprinkler?', False)]))

    # print(lecture_Example_2.map_and_mpe(order_function=Heuristics.MIN_ORDER, e=[('O', False), ('J', False)], M=['I',  'Y']))
    # print(lecture_Example_2.map_and_mpe(order_function=Heuristics.MIN_FILL, e=[('O', False), ('J', False)], M=['I',  'Y']))
    # print(lecture_Example_2.map_and_mpe(order_function=Heuristics.RANDOM, e=[('O', False), ('J', False)], M=['I', 'Y']))
    # bn = BayesNet()
    # bn.load_from_bifxml("testing/lecture_Example2.BIFXML")
    # interaction_graph = bn.get_interaction_graph()
    # bn.draw_graph(interaction_graph)


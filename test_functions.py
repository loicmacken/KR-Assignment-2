from typing import List, Dict
from BNReasoner import BNReasoner
from BayesNet import BayesNet
import json   
import copy
   
class Tester:
    """
    Contains the functionality for testing each of the functions of BNReasoner.py by comparing with pre-calculated values (by hand)
    """
    def __init__(self, reasoner: BNReasoner, data: Dict) -> None:
        self.reasoner = reasoner
        self.data = data

    def test_d_sep(self) -> None:   
        for X, Y, Z, result in self.data['d_sep']:
            assert self.reasoner.d_seperation(X, Y, Z) == result

    def test_min_degree(self) -> None:
        for X, result in self.data['min_degree']:
            assert self.reasoner.ordering_min_degree(X) == result
        
    def test_min_fill(self) -> None:
        for X, result in self.data['min_fill']:
            assert self.reasoner.ordering_min_fill(X) == result

    def test_net_prune(self) -> None:
        for Q, e, vars, edges in self.data['net_prune']:
            temp_bn = copy.deepcopy(self.reasoner.bn)
            assert isinstance(temp_bn, BayesNet)

            edge_list: List[tuple] = list(tuple(x) for x in edges)
            
            temp_bn.net_prune(set(Q), e)
            # verify whether the pruned network has the same variables as the computed test data
            assert temp_bn.get_all_variables() == vars
            assert temp_bn.get_all_edges() == edge_list

    def test_margin_dist(self) -> None:
        for Q, e, pi, results in self.data['marginal_distrib']:
            e_list: List[tuple] = list(tuple(x) for x in e)

            # the output CPT from the marginal distribution
            df = self.reasoner.marginal_dist(Q, e_list, pi)  # type: ignore

            # the column 'p' which has the probabilities
            outputs = list(df['p'])

            for out, res in zip(outputs, results):
                # test whether the probability value of the last row is within a delta margin of the test value
                assert (float(res) - DELTA) < float(out) < (float(res) + DELTA)

    def test_map_mpe(self) -> None:
        for _ in self.data['map_and_mpe']:
            pass

if __name__ =='__main__':
    test_problems = ['dog_problem', 'lecture_example', 'lecture_example2']

    # delta value for accuracy in floating point outputs, i.e. the output will have to be within +/- one delta of the desired result
    DELTA = 0.01

    for prob in test_problems:
        # create a reasoner object based on the problem BIFXML file
        reasoner = BNReasoner('testing/' + prob + '.BIFXML')

        # get test data from json
        data = {}
        with open('test_data/' + prob + '.json', 'r') as infile:
            data = json.load(infile)

        # create a tester object based on the reasoner and the data
        tester = Tester(reasoner, data)

        # test d-seperation
        tester.test_d_sep()

        # test min-degree ordering
        tester.test_min_degree()

        # test min-fill ordering
        tester.test_min_fill()

        # test network pruning
        tester.test_net_prune()

        # test marginal distributions
        tester.test_margin_dist()

        # test MAP and MPE
        tester.test_map_mpe()
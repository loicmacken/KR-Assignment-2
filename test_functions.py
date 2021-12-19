from typing import List, Tuple
from BNReasoner import BNReasoner
import json   
import copy
   

def test_d_sep(data, reasoner):   
    for X, Y, Z, result in data['d_sep']:
        assert reasoner.d_seperation(X, Y, Z) == result

def test_min_degree(data, reasoner):
    for X, result in data['min_degree']:
        assert reasoner.ordering_min_degree(X) == result
    
def test_min_fill(data, reasoner):
    for X, result in data['min_fill']:
        assert reasoner.ordering_min_fill(X) == result

def test_net_prune(data, reasoner):
    for Q, e, vars, edges in data['net_prune']:
        temp_bn = copy.deepcopy(reasoner.bn)
        assert isinstance(temp_bn, BayesNet)

        e_list: List[Tuple] = list(tuple(x) for x in e)
        edge_list: List[Tuple] = list(tuple(x) for x in edges)
        
        temp_bn.net_prune(set(Q), e)
        # verify whether the pruned network has the same variables as the computed test data
        assert temp_bn.get_all_variables() == vars
        assert temp_bn.get_all_edges() == edge_list

def test_margin_dist(data, reasoner):
    for Q, e, pi, result in data['marginal_distrib']:
        e_list: List[Tuple] = list(tuple(x) for x in e)

        # the output CPT from the marginal distribution
        df = reasoner.marginal_dist(Q, e_list, pi)  # type: ignore

        # the last row, which is where all variables are true
        output = df.iloc[-1].loc['p']

        result_float = float(result[0])

        # test whether the probability value of the last row is within a delta margin of the test value
        assert (result_float - DELTA) < output < (result_float + DELTA)

def test_map_mpe(data):
    for _ in data['map_and_mpe']:
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

        # test d-seperation
        test_d_sep(data, reasoner)

        # test min-degree ordering
        test_min_degree(data, reasoner)

        # test min-fill ordering
        test_min_fill(data, reasoner)

        # test network pruning
        test_net_prune(data, reasoner)

        # test marginal distributions
        test_margin_dist(data, reasoner)

        # test MAP and MPE
        test_map_mpe(data, reasoner)
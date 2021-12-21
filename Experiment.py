import helper_functions as helpers
from BNReasoner import BNReasoner, Heuristics
from BayesNet import BayesNet
from enum import Enum
import random
import time

class Queries(Enum):
    MAP='MAP'
    MPE='MPE'

runs = 10
ratio_evidence = 2.5
ratio_map_vars = 5

def load_bns() -> dict:
    bns = {}
    start_num_nodes = 5
    start_num_roots = 2
    start_num_min_edges = 1
    start_num_max_edges = 3
    increase_nodes = 10
    num_bns = 5

    for id in range(num_bns):
        bn = BayesNet()
        data = bn.generate_random(number=start_num_nodes, 
            n_roots=start_num_roots, max_edges=start_num_max_edges, 
            min_edges=start_num_min_edges)
        bns.update({id: (data, bn)})
        start_num_nodes = start_num_nodes + increase_nodes

    return bns
            
def run_experiment():
    bns = load_bns()

    for run_number in range(runs):
        for data, bn in bns.values():
            e=[]
            reasoner = BNReasoner(bn)
            nodes = bn.get_all_variables()
            start_num_evidence = int(len(nodes)/ratio_evidence)
            start_num_map_vars = int(len(nodes)/ratio_map_vars)
            selected_evidence = random.sample(nodes, k=start_num_evidence)
            selected_map_variables = random.sample(set(nodes)-set(selected_evidence), k=start_num_map_vars)
            
            for evidence in selected_evidence:
                e.append((evidence, True))

            for method in Heuristics:
                print(method)

                # mpe and map
                for query in Queries:
                    if query is Queries.MPE: map_variables = []
                    else: map_variables = selected_map_variables

                    start_time = time.time()
                    result = reasoner.map_and_mpe(order_function=method, e=e, M=map_variables)
                    end_time = time.time() - start_time

                    data['n_evidence'] = start_num_evidence
                    data['time'] = end_time
                    data['degrees_occurred'] = result[0]
                    data['result'] = result[1]
                    helpers.save_data(query.value, method.value, run_number, data)

def create_experiment_graphs(query):
    data = helpers.combine_results(query, Heuristics)
    helpers.create_lineplots(data, query, Heuristics)
    helpers.create_barchart(data, query, Heuristics)


if __name__ =='__main__':
    # run_experiment()
    create_experiment_graphs(Queries.MPE.value)
    create_experiment_graphs(Queries.MAP.value)


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
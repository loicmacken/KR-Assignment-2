import helper_functions as helpers
from BNReasoner import BNReasoner, Heuristics
from BayesNet import BayesNet
import random
import time


run_number = 2
num_bns = 5
num_categories = 5
start_num_evidence = 2

def load_bns() -> dict:
    bns = {}
    start_num_nodes = 5
    start_num_roots = 2
    start_num_min_edges = 2
    start_num_max_edges = 4
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

    # for run_number in range(run_number):
    for data, bn in bns.values(): 
        reasoner = BNReasoner(bn)
        for method in Heuristics:
            print(method)
            e=[]
            nodes = bn.get_all_variables()
            start_num_evidence = int(len(nodes)/2.5)
            selected_evidence = random.sample(nodes, start_num_evidence)
            for evidence in selected_evidence:
                e.append((evidence, True))
            
            start_time = time.time()
            #mpe
            result = reasoner.map_and_mpe(order_function=method, e=e)
            end_time = time.time() - start_time

            data['n_evidence'] = start_num_evidence
            data['time'] = end_time
            data['result'] = result
            helpers.save_data(method.value, run_number, data)

def create_experiment_graphs():
    data = helpers.combine_results(Heuristics, '')
    helpers.create_lineplots(data, '', Heuristics, x_val="n_nodes", y_val="time")
    pass


if __name__ =='__main__':
    run_experiment()
    create_experiment_graphs()

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
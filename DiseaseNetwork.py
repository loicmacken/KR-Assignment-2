import copy

from BNReasoner import BNReasoner, Heuristics
import pandas as pd

FILE_PATH = "./testing/BN_DISEASES_3.BIFXML"

class DiseaseNetwork():
    # load data
    bn = BNReasoner(FILE_PATH)     

    # draw graphs
    # bn.draw_structure()
    # bn.draw_interaction_graph()

    # d_sep query
    def get_d_sep(self):
        return self.bn.d_seperation(X=['G', 'JT'], Y=['CHD', 'LC'], Z=['OS', 'M'])
        # Assume we want to test if set of variables G, J are d-separated from CHD, LC (Lung Cancer) by (evidence OS, M
        pass

    # priori marginal query
    def get_priori_marginal(self):
        return self.bn.marginal_dist(Q=['CHD', 'G'], e=[], order_function=Heuristics.MIN_ORDER)
        # Pr(CHD (Coronary Heart Disease ∧ G (Gender)
        pass

    # posterior marginal query.
    def get_posterior_marginal(self):
        #TODO normalize with evidence
        return self.bn.marginal_dist(Q=['CHD'], e=[('S', True), ('JB', False)], order_function = Heuristics.MIN_ORDER)
        # Pr(CHD = True | S (Smoking) = True, JB (Job Burnout) = False
        pass

    # MAP query
    def get_map(self) -> pd.DataFrame:
        bn = copy.copy(self.bn)
        return bn.map_and_mpe(order_function=Heuristics.MIN_ORDER, e=[('OS', True), ('RSO', False)], M=['G',  'JT'])

    # MPE query
    def get_mpe(self) -> pd.DataFrame:
        bn = copy.copy(self.bn)
        return bn.map_and_mpe(order_function=Heuristics.MIN_FILL, e=[('G', False), ('JT', True)])


if __name__ =='__main__':
    network = DiseaseNetwork()
    #print(network.get_mpe())
    # print(network.get_map()) # FIX
    #print(network.get_posterior_marginal())
    print(network.get_d_sep()) # networkFIX
    #print(network.get_priori_marginal())
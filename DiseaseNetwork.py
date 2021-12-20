from BNReasoner import BNReasoner, Heuristics
import pandas as pd

FILE_PATH = "testing/diseases.BIFXML"

class DiseaseNetwork():
    # load data
    bn = BNReasoner(FILE_PATH)     

    # draw graphs
    bn.draw_structure()
    bn.draw_interaction_graph()

    # d_sep query
    def get_d_sep(self):
        # return bn.d_sep()
        pass

    # priori marginal query
    def get_priori_marginal(self):
        # return bn.marginal_dist()
        pass

    # posterior marginal query.
    def get_posterior_marginal(self):
        #TODO normalize with evidence
        # return self.bn.marginal_dist()
        pass

    # MAP query
    def get_map(self):
        # print(bn.map_and_mpe(order_function=Heuristics.MIN_ORDER, e=[('O', False), ('J', False)], M=['I',  'Y']))
        pass

    # MPE query
    def get_mpe(self) -> pd.DataFrame:
        return self.bn.map_and_mpe(order_function=Heuristics.MIN_ORDER, e=[('gender', False), ('job', True)])


if __name__ =='__main__':
    network = DiseaseNetwork()
    network.get_mpe()
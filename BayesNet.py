from typing import List, Tuple, Dict, Set
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.readwrite import XMLBIFReader
import math
import itertools
import pandas as pd
from copy import deepcopy


class BayesNet:

    def __init__(self) -> None:
        # initialize graph structure
        self.structure = nx.DiGraph()

    # LOADING FUNCTIONS ------------------------------------------------------------------------------------------------
    def create_bn(self, variables: List[str], edges: List[Tuple[str, str]], cpts: Dict[str, pd.DataFrame]) -> None:
        """
        Creates the BN according to the python objects passed in.
        
        :param variables: List of names of the variables.
        :param edges: List of the directed edges.
        :param cpts: Dictionary of conditional probability tables.
        """
        # add nodes
        [self.add_var(v, cpt=cpts[v]) for v in variables]

        # add edges
        [self.add_edge(e) for e in edges]

        # check for cycles
        if not nx.is_directed_acyclic_graph(self.structure):
            raise Exception('The provided graph is not acyclic.')

    def load_from_bifxml(self, file_path: str) -> None:
        """
        Load a BayesNet from a file in BIFXML file format. See description of BIFXML here:
        http://www.cs.cmu.edu/afs/cs/user/fgcozman/www/Research/InterchangeFormat/

        :param file_path: Path to the BIFXML file.
        """
        # Read and parse the bifxml file
        with open(file_path) as f:
            bn_file = f.read()
        bif_reader = XMLBIFReader(string=bn_file)

        # load cpts
        cpts = {}
        # iterating through vars
        for key, values in bif_reader.get_values().items():
            values = values.transpose().flatten()
            n_vars = int(math.log2(len(values)))
            worlds = [list(i) for i in itertools.product([False, True], repeat=n_vars)]
            # create empty array
            cpt = []
            # iterating through worlds within a variable
            for i in range(len(values)):
                # add the probability to each possible world
                worlds[i].append(values[i])
                cpt.append(worlds[i])

            # determine column names
            columns = bif_reader.get_parents()[key]
            columns.reverse()
            columns.append(key)
            columns.append('p')
            cpts[key] = pd.DataFrame(cpt, columns=columns)
        
        # load vars
        variables = bif_reader.get_variables()
        
        # load edges
        edges = bif_reader.get_edges()

        self.create_bn(variables, edges, cpts)

    # METHODS THAT MIGHT ME USEFUL -------------------------------------------------------------------------------------

    def get_children(self, variable: str) -> List[str]:
        """
        Returns the children of the variable in the graph.
        :param variable: Variable to get the children from
        :return: List of children
        """
        return [c for c in self.structure.successors(variable)]

    def get_cpt(self, variable: str) -> pd.DataFrame:
        """
        Returns the conditional probability table of a variable in the BN.
        :param variable: Variable of which the CPT should be returned.
        :return: Conditional probability table of 'variable' as a pandas DataFrame.
        """
        try:
            return self.structure.nodes[variable]['cpt']
        except KeyError:
            raise Exception('Variable not in the BN')

    def get_all_variables(self) -> List[str]:
        """
        Returns a list of all variables in the structure.
        :return: list of all variables.
        """
        return [n for n in self.structure.nodes]

    def get_all_cpts(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary of all cps in the network indexed by the variable they belong to.
        :return: Dictionary of all CPTs
        """
        cpts = {}
        for var in self.get_all_variables():
            cpts[var] = self.get_cpt(var)

        return cpts

    def get_interaction_graph(self):
        """
        Returns a networkx.Graph as interaction graph of the current BN.
        :return: The interaction graph based on the factors of the current BN.
        """
        # Create the graph and add all variables
        int_graph = nx.Graph()
        [int_graph.add_node(var) for var in self.get_all_variables()]

        # connect all variables with an edge which are mentioned in a CPT together
        for var in self.get_all_variables():
            involved_vars = list(self.get_cpt(var).columns)[:-1]
            for i in range(len(involved_vars)-1):
                for j in range(i+1, len(involved_vars)):
                    if not int_graph.has_edge(involved_vars[i], involved_vars[j]):
                        int_graph.add_edge(involved_vars[i], involved_vars[j])
        return int_graph

    @staticmethod
    def get_compatible_instantiations_table(instantiation: pd.Series, cpt: pd.DataFrame):
        """
        Get all the entries of a CPT which are compatible with the instantiation.

        :param instantiation: a series of assignments as tuples. E.g.: pd.Series(("A", True), ("B", False))
        :param cpt: cpt to be filtered
        :return: table with compatible instantiations and their probability value
        """
        var_names = instantiation.index.values
        var_names = [v for v in var_names if v in cpt.columns]  # get rid of excess variables names
        compat_indices = cpt[var_names] == instantiation[var_names].values
        compat_indices = [all(x[1]) for x in compat_indices.iterrows()]
        compat_instances = cpt.loc[compat_indices]
        return compat_instances

    def update_cpt(self, variable: str, cpt: pd.DataFrame) -> None:
        """
        Replace the conditional probability table of a variable.
        :param variable: Variable to be modified
        :param cpt: new CPT
        """
        self.structure.nodes[variable]["cpt"] = cpt

    @staticmethod
    def reduce_factor(instantiation: pd.Series, cpt: pd.DataFrame) -> pd.DataFrame:
        """
        Creates and returns a new factor in which all probabilities which are incompatible with the instantiation
        passed to the method to 0.

        :param instantiation: a series of assignments as tuples. E.g.: pd.Series({"A", True}, {"B", False})
        :param cpt: cpt to be reduced
        :return: cpt with their original probability value and zero probability for incompatible instantiations
        """
        var_names = instantiation.index.values
        var_names = [v for v in var_names if v in cpt.columns]  # get rid of excess variables names
        if len(var_names) > 0:  # only reduce the factor if the evidence appears in it
            new_cpt = deepcopy(cpt)
            incompat_indices = cpt[var_names] != instantiation[var_names].values
            incompat_indices = [any(x[1]) for x in incompat_indices.iterrows()]
            new_cpt.loc[incompat_indices, 'p'] = 0.0
            return new_cpt
        else:
            return cpt

    def draw_structure(self) -> None:
        """
        Visualize structure of the BN.
        """
        nx.draw(self.structure, with_labels=True, node_size=3000)
        plt.show()

    # BASIC HOUSEKEEPING METHODS ---------------------------------------------------------------------------------------

    def add_var(self, variable: str, cpt: pd.DataFrame) -> None:
        """
        Add a variable to the BN.
        :param variable: variable to be added.
        :param cpt: conditional probability table of the variable.
        """
        if variable in self.structure.nodes:
            raise Exception('Variable already exists.')
        else:
            self.structure.add_node(variable, cpt=cpt)

    def add_edge(self, edge: Tuple[str, str]) -> None:
        """
        Add a directed edge to the BN.
        :param edge: Tuple of the directed edge to be added (e.g. ('A', 'B')).
        :raises Exception: If added edge introduces a cycle in the structure.
        """
        if edge in self.structure.edges:
            raise Exception('Edge already exists.')
        else:
            self.structure.add_edge(edge[0], edge[1])

        # check for cycles
        if not nx.is_directed_acyclic_graph(self.structure):
            self.structure.remove_edge(edge[0], edge[1])
            raise ValueError('Edge would make graph cyclic.')

    def del_var(self, variable: str) -> None:
        """
        Delete a variable from the BN.
        :param variable: Variable to be deleted.
        """
        self.structure.remove_node(variable)

    def del_edge(self, edge: Tuple[str, str]) -> None:
        """
        Delete an edge form the structure of the BN.
        :param edge: Edge to be deleted (e.g. ('A', 'B')).
        """
        self.structure.remove_edge(edge[0], edge[1])

    # ADDED FUNCTIONS -----------------------------------

    def draw_graph(self, graph: nx.DiGraph) -> None:
        """
        Visualize structure of the BN.
        """
        nx.draw(graph, with_labels=True, node_size=3000)
        plt.show()

    def get_all_edges(self) -> List[Tuple[str, str]]:
        """
        """
        return [e for e in self.structure.edges]

    def get_leaf_nodes(self) -> List[str]:
        """
        """
        vars: Set[str] = set(self.get_all_variables())

        for node, _ in self.get_all_edges():
            if node in vars:
                vars.remove(node)

        return list(vars)

    def get_root_nodes(self) -> List[str]:
        """
        """
        vars: Set[str] = set(self.get_all_variables())

        for _, node in self.get_all_edges():
            if node in vars:
                vars.remove(node)

        return list(vars)

    def get_node_parents(self, W: str) -> List[str]:
        """
        """
        parents: Set[str] = set()
        for parent, child in self.get_all_edges():
            if child == W:
                parents.add(parent)

        return list(parents)

    def is_connected(self, X: Set[str], Y: Set[str]) -> bool:
        """
        """
        visited: Set[str] = set()

        for var in X:
            if self._get_connections(var, Y, visited):
                return True
        return False

    def _get_connections(self, var: str, Y: Set[str], visited: Set[str]) -> bool:
        """
        """
        if var in Y: return True
        
        visited.add(var)
        edges: Set[str] = set(self.get_node_parents(var)) | set(self.get_children(var)) - visited

        for edge in edges:
            return self._get_connections(edge, Y, visited)
        return False

    def prune_leaves(self, vars: Set[str]) -> None:
        while(True):
            leaves: Set[str] = set(self.get_leaf_nodes())
            leaves_to_delete = leaves - vars

            if not leaves_to_delete:
                return

            for leaf in leaves_to_delete:
                self.del_var(leaf)

    def d_sep(self, X: Set[str], Y: Set[str], Z: Set[str]) -> bool:
        """
        Gets the d-seperation given the three sets of variables: X, Y and Z,
        i.e. returns true if X is independent of Y given Z in the self.bn

        :param X: variable set X
        :param Y: variable set Y
        :param Z: variable set Z

        :return: dsep(X, Y, Z)
        """
        XYZ: Set[str] = X | Y | Z

        # delete leaf nodes W not part of X U Y U Z
        self.prune_leaves(XYZ)

        # delete all edges outgoing from Z
        for var in Z:
            for child in self.get_children(var):
                self.del_edge((var, child))

        # moralize by marrying parents (making an edge between them)
        for var in self.get_all_variables():
            parents = self.get_node_parents(var)
            if len(parents) > 1:
                for i in range(len(parents) - 1):
                    self.add_edge((parents[i], parents[i+1]))
        
        # remove givens: delete all edges and nodes of Z
        for var in Z:
            for parent in self.get_node_parents(var):
                self.del_edge((parent, var))
            self.del_var(var)

        # d-seperated if X and Y are NOT connected
        return not self.is_connected(X, Y)

    def min_degree(self, X: List[str]) -> List[str]:
        """
        """
        G = self.get_interaction_graph()
        pi = []

        for i in range(len(X)):
            var = ''
            min_val = 1000
            for n, nbrdict in G.adjacency():
                if len(nbrdict) < min_val:
                    var = n

            neighbors = list(G.neighbors(var))
            for node in neighbors:
                for neighbor in neighbors:
                    if node == neighbor:
                        continue
                    if not (G.has_edge(node, neighbor) or G.has_edge(neighbor, node)):
                        G.add_edge(node, neighbor)

            pi.append(var)

            G.remove_node(var)
            X.remove(var)

        return pi

    def min_fill(self, X: List[str]) -> List[str]:
        """
        """
        G = self.get_interaction_graph()
        pi = []

        for i in range(len(X)):
            min_var = ''
            min_edges = 1000
            for var in X:
                edges = 0
                neighbors = list(G.neighbors(var))
                for node in neighbors:
                    for neighbor in neighbors:
                        if node == neighbor:
                            continue
                        if not (G.has_edge(node, neighbor) or G.has_edge(neighbor, node)):
                            edges += 1
                if edges < min_edges:
                    min_var = var
                    min_edges = edges

            neighbors = list(G.neighbors(min_var))
            for node in neighbors:
                for neighbor in neighbors:
                    if node == neighbor:
                        continue
                    if not (G.has_edge(node, neighbor) or G.has_edge(neighbor, node)):
                        G.add_edge(node, neighbor)

            pi.append(min_var)

            G.remove_node(min_var)
            X.remove(min_var)

        return pi
                
    def net_prune(self, Q: Set[str], e: List[tuple[str, bool]]) -> None:
        """
        Node- and edgeprunes the network self.bn
        Based on query variables Q and evidence e

        :param Q: set of query variables
        :param e: list of evidence
        """
        E = set([var for var, _ in e])
        QE = Q | E

        # first prune the leaves not in Q U E
        self.prune_leaves(QE)

        # then prune edges outgoing from E
        for parent, child in self.get_all_edges():
            if parent in E:
                cpt = self.get_cpt(child)
                cpt = self.get_compatible_instantiations_table(pd.Series(e), cpt)
                self.update_cpt(child, cpt)

    def marginal_dist(self, Q: List[str], pi: List[str], e: List[tuple[str, bool]]): # TODO: add types
        """
        Computes the marginal distribution P(Q|e)
        given the query variables Q and possibly empty evidence e

        :param Q: query variables
        :param pi: ordering of network variables not in Q
        :param e: evidence

        :return: TODO
        """
        cpts = self.get_all_cpts()
        for c in cpts:
            print(cpts)

        # update CTPs based on evidence, if it's not empty
        if e:
            for var in pi:
                cpt = self.get_cpt(var)
                cpt = self.get_compatible_instantiations_table(pd.Series(e), cpt)
                self.update_cpt(var, cpt)

        cpts = self.get_all_cpts()
        for c in cpts:
            print(cpts)

        
            


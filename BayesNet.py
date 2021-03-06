from typing import List, Tuple, Dict, Set, Callable
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.readwrite import XMLBIFReader
import math
import itertools
import random
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

    # GENERATE BN ------------------------------------------------------------------------------------------------
    def generate_random(self, number: int=10, n_roots: int=2, max_edges: int=4, min_edges:int=2) -> Dict:
        """
        Generate a BayesNet 
        """
        # create nodes
        variables = []
        for i in range(number):
            variables.append(str(i))

        # create edges and create cpts
        edges = []
        cpts = {}
        neighbors = {}
        degree = 0
        index = len(variables) - 1
        for node in reversed(variables):
            vars=[]
            if node not in neighbors: neighbors[node] = set()
            nodes=variables[:index]
            selected_nodes=[]

            if index > n_roots:
                start = min_edges
                k = start
                max = round(int(index/2))
                if max > max_edges: max=max_edges
                if max > start: k=random.randint(start, max)
                selected_nodes = random.sample(nodes, k)
            if index == n_roots:
                selected_nodes = nodes
            
            # update neighbors
            neighbors[node].update(set(selected_nodes))

            for parent in selected_nodes:
                edges.append([parent, node])
                vars.append(parent)

                # set interaction neighbors
                if parent not in neighbors: neighbors[parent] = set()
                set_neighbors = selected_nodes + [node]
                set_neighbors.remove(parent) 
                neighbors[parent].update(set(set_neighbors))

            # find degree
            if len(neighbors[node]) > degree:
                degree = len(neighbors[node])
            vars.append(node)

            worlds = [list(i) for i in itertools.product([False, True], repeat=len(vars))]
            columns = list(vars)
            df = pd.DataFrame(worlds, columns=columns)            

            # iterating through worlds within a variable
            cpt = []
            n_vars = 2**(len(vars)-1)
            for i in range(n_vars):
                # add the probability to each possible world
                p = round(random.uniform(0,1), 2)
                p_reverse = round(1-p, 2)
                cpt.append(p)
                cpt.append(p_reverse)

            df.insert(loc=len(df.columns), column='p', value=cpt)
            cpts.update({node: df})

            index-=1

        self.create_bn(variables, edges, cpts) # type: ignore

        return {'n_nodes': number, 'degree': degree, 'n_roots': n_roots, 'max_edges': max_edges, 'min_edges': min_edges}

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
            values = values.transpose().flatten() # type: ignore
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

        self.create_bn(variables, edges, cpts) # type: ignore

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
        compat_indices = [all(x[1]) for x in compat_indices.iterrows()] # type: ignore
        compat_instances = cpt.loc[compat_indices]
        return compat_instances.reset_index(drop=True)

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
            incompat_indices = [any(x[1]) for x in incompat_indices.iterrows()] # type: ignore
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

    def draw_structure_sample(self) -> None:
        """
        Visualize structure of the BN.
        """
        nx.draw(self.structure, with_labels=True, node_size=200)
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

    @staticmethod
    def draw_graph(graph: nx.Graph) -> None:
        """
        Visualize structure of the BN.

        :param graph: the graph to visualize
        """
        nx.draw(graph, with_labels=True, node_size=3000)
        plt.show()

    def get_all_edges(self) -> List[Tuple[str, str]]:
        """
        Returns all edges in the network

        :return: list of edges in the BN
        """
        return [e for e in self.structure.edges]

    def get_leaf_nodes(self) -> List[str]:
        """
        Returns all leaf nodes in the network

        :return: list of leaf nodes in the BN
        """
        vars: Set[str] = set(self.get_all_variables())

        for node, _ in self.get_all_edges():
            if node in vars:
                vars.remove(node)

        return list(vars)

    def get_root_nodes(self) -> List[str]:
        """
        Returns all root nodes in the network

        :return: list of root nodes in the BN
        """
        vars: Set[str] = set(self.get_all_variables())

        for _, node in self.get_all_edges():
            if node in vars:
                vars.remove(node)

        return list(vars)

    def get_node_parents(self, W: str) -> List[str]:
        """
        Returns the parents of the given node W

        :param W: variable of which the parents will be returned

        :return: list of parents of W
        """
        parents: Set[str] = set()
        for parent, child in self.get_all_edges():
            if child == W:
                parents.add(parent)

        return list(parents)

    def _is_connected(self, X: Set[str], Y: Set[str]) -> bool:
        """
        Recursively verifies whether there is a path leading from nodes in X to nodes in Y,
        disregarding any directionality.

        :param X: the set of nodes to find connections from
        :param Y: the set of nodes to find connections to

        :return: True if connected
        """
        visited: Set[str] = set()

        for var in X:
            if self._get_connections(var, Y, visited):
                return True
        return False

    def _get_connections(self, var: str, Y: Set[str], visited: Set[str]) -> bool:
        """
        Helper function for recursion of _is_connected.

        :param var: the variable to start from
        :param Y: the set of variables to find a connection to
        :param visited: the nodes that have been visited previously by this function

        :return: True if var is connected to a variable in Y
        """
        if var in Y: return True
        
        visited.add(var)
        edges: Set[str] = set(self.get_node_parents(var)) | set(self.get_children(var)) - visited

        for edge in edges:
            if self._get_connections(edge, Y, visited):
                return True
        return False

    def _prune_leaves(self, vars: Set[str]) -> None:
        """
        Iteratively removes any leaf nodes that are not in the set of given nodes.
        
        :param vars: the set of nodes that will not be removed
        """
        while(True):
            leaves: Set[str] = set(self.get_leaf_nodes())
            # set difference of leaves and vars, i.e. the leaves that are not in vars
            leaves_to_delete = leaves - vars

            # no more leaves to delete: return
            if not leaves_to_delete:
                return

            # delete leaf
            for leaf in leaves_to_delete:
                self.del_var(leaf)

    def _get_cpts_x(self, X: str, cpts: Dict[str, pd.DataFrame]) -> tuple[List[pd.DataFrame], List[str]]:
        """
        Returns all CPTs involving the variable X

        :param X: a variable in the BN to retrieve CPTs from
        :param cpts: list of all CPTs 

        :return: a list of only the CPTs that include X
        """
        cpts_x = []
        cpts_x_values = []

        for key in cpts:
            if X in list(cpts[key].columns):
                cpts_x.append(cpts[key])
                cpts_x_values.append(key)

        return cpts_x, cpts_x_values

    def _replace_cpts(self, cpts: Dict[str, pd.DataFrame], remove: List[str], add: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Delete all CPTs involving input CPT keys and add CPTs from input list

        :param remove: List of CPT keys to be replaced
        :param add: List of CPTs to add

        :return: the updated dict of CPTs
        """
        for variable in remove:
            del cpts[variable]

        cpts.update(add)

        return cpts

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
        self._prune_leaves(XYZ)

        # delete all edges outgoing from Z
        for var in Z:
            for child in self.get_children(var):
                self.del_edge((var, child))

        # d-seperated if X and Y are NOT connected
        return not self._is_connected(X, Y)

    def random_order(self, X: List[str]) -> List[str]:
        """
        Random order heuristic, returns a random order of the input list of variables.
        
        :param X: input variable list

        :return: list X rearranged in a random order
        """
        random.shuffle(X)

        return X

    def min_degree(self, X: List[str]) -> List[str]:
        """
        Gets the ordering of the variable list X
        based on min-degree heuristics.

        :param X: variables in network N

        :return: an ordering PI of variables X
        """
        G = self.get_interaction_graph()
        pi = []

        X_copy = X.copy()
        len_X = len(X_copy)

        for i in range(len_X):
            var = ''
            min_val = 1000

            for n in X_copy:
                nodes = len(list(G.neighbors(n)))
                if nodes < min_val:
                    var = n
                    min_val = nodes

            # add an edge between every pair of non-adjacent neighbors of pi in G
            neighbors = list(G.neighbors(var))
            for node in neighbors:
                for neighbor in neighbors:
                    if node == neighbor:
                        continue
                    if not (G.has_edge(node, neighbor) or G.has_edge(neighbor, node)):
                        G.add_edge(node, neighbor)

            pi.append(var)

            # delete variable pi from G and from X
            G.remove_node(var)
            X_copy.remove(var)

        return pi

    def min_fill(self, X: List[str]) -> List[str]:
        """
        Gets the ordering of the variable list X
        based on min-fill heuristics.

        :param X: variable list X

        :return: an ordering PI of variables X
        """
        G = self.get_interaction_graph()
        pi = []

        X_copy = X.copy()
        len_X = len(X_copy)

        for i in range(len_X):
            min_var = ['', []]
            min_edges = 1000

            # find variable in X with smallest number of added edges in G
            for var in X_copy:
                edges = []
                neighbors = list(G.neighbors(var))

                for node in neighbors:
                    for neighbor in neighbors:
                        if node == neighbor or [neighbor, node] in edges:
                            continue
                        if not (G.has_edge(node, neighbor) or G.has_edge(neighbor, node)):
                            edges.append([node, neighbor])

                if len(edges) < min_edges:
                    min_var = [var, edges]
                    min_edges = len(edges)
            
            # add an edge between every pair of non-adjacent neighbors of pi in G
            for node, neighbor in min_var[1]:
                G.add_edge(node, neighbor)

            pi.append(min_var[0])

            # delete variable pi from G and from X
            G.remove_node(min_var[0])
            X_copy.remove(min_var[0])

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

        evidence = dict(e)

        # first prune the leaves not in Q U E
        self._prune_leaves(QE)

        # then prune edges outgoing from E
        for parent, child in self.get_all_edges():
            if parent in E:
                # update CPT
                instantiation = pd.Series({parent: evidence[parent]})
                cpt = self.get_cpt(child)
                new_cpt = self.get_compatible_instantiations_table(instantiation, cpt)
                self.update_cpt(child, new_cpt)

                # remove edge
                self.del_edge((parent, child))

    def _create_factor(self, X: Set[str], value: List):
        """
        Creates a factor over the cpt where all the probabilities of X variables are equal to zero.

        :param X: a set of variables to set to zero
        :param value: added cpt value

        :return: a factor over variables X that are all equal to zero
        """
        worlds = [list(i) + value for i in itertools.product([False, True], repeat=len(X))]
        columns = list(X) + ['new_p']
        cpt = pd.DataFrame(worlds, columns=columns)
        
        return cpt

    def _sum_out(self, f_x: pd.DataFrame, Z: str) -> pd.DataFrame:
        """
        Sums out the variables Z from the factor f_x

        :param f_x: a factor over variables X
        :param Z: a subset of variables X

        :return: a factor corresponding to ??_z(f)
        """
        X = set(f_x.columns.tolist()) - set(['p']) - set(['instantiations'])
        Y = X - set([Z]) # e.g BC

        # check if there are any variables left, if not, return an empty frame
        if not Y: return pd.DataFrame()

        new_f: pd.DataFrame = f_x.groupby(list(Y), as_index = False)['p'].sum()

        return new_f

    def _mult_factors(self, factors: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Factors the input list

        :param factors: list of factor to multiply

        :return: a factor corresponding to the product ??(f)
        """
        if len(factors) == 1:
            return factors[0]

        Z = set()
        for f in factors:
            X = set(f.columns.tolist()) - set(['p']) - set(['instantiations'])
            Z = Z | X

        new_f = self._create_factor(Z, value=[1])
        new_f['new_instantiations'] = ' '

        for f1 in factors:
            new_f = new_f.merge(f1)
            new_f.new_p = new_f.new_p * new_f.p
            new_f.drop('p', axis=1, inplace=True)
            if 'instantiations' in f1.columns:
                new_f['new_instantiations'] = new_f['instantiations'].astype(str) + new_f['new_instantiations']
                new_f.drop('instantiations', axis=1, inplace=True)

        new_f.rename(columns={'new_instantiations':'instantiations'}, inplace=True)
        new_f.rename(columns={'new_p':'p'}, inplace=True)

        return new_f

    def _max_out(self, f_x: pd.DataFrame, Z: str, visited: set) -> pd.DataFrame: 
        """
        Maxes out the variables Z from factor f_x

        :param f_x: a factor over variables X
        :param Z: a subset of variables X

        :return: a factor corresponding to Max_z(f)
        """
        X = set(f_x.columns.tolist()) - set(['p']) - set(['instantiations'])
        Y = X - set([Z]) # e.g BC
        new_f = pd.DataFrame()
        if 'instantiations' not in f_x.columns: f_x['instantiations'] = ''

        if len(Y) > 0:
            new_f: pd.DataFrame = f_x.loc[f_x.groupby(list(Y))['p'].idxmax()]
            new_f['instantiations'] = Z + '=' + new_f[Z].astype(str) + ',' + f_x['instantiations']
            new_f.drop(Z, axis=1, inplace=True)
        else:
            #TODO pick the highest value
            max = f_x['p'].max()
            new_f: pd.DataFrame = f_x[f_x['p'] == max]
            new_f['instantiations'] = Z + '=' + new_f[Z].astype(str) + ',' + f_x['instantiations']
            new_f.drop(Z, axis=1, inplace=True)

        return new_f

    def marginal_distrib(self, Q: List[str],  e: List[tuple[str, bool] or None], pi: List[str]) -> pd.DataFrame:
        """
        Computes the marginal distribution P(Q|e)
        given the query variables Q, possibly empty evidence e and ordering pi

        :param Q: query variables
        :param e: evidence
        :param pi: ordering of network variables not in Q

        :return: the marginal distribution of P(Q, E) normalized by the E or P(Q) if E is empty
        """
        cpts = self.get_all_cpts()
        cpts_e = {}
        f_sum = pd.DataFrame()
    
        for key in cpts:
            cpts_e.update({key: self.reduce_factor(pd.Series(dict(e)), cpts[key])})
        
        for i in range(len(pi)):
            f_pi, fpi_keys = self._get_cpts_x(pi[i], cpts_e)
            f_mult = self._mult_factors(f_pi)
            f_sum = self._sum_out(f_mult, pi[i])
            cpts_e = self._replace_cpts(cpts_e, fpi_keys, {pi[i]: f_sum})

        marginals = self._mult_factors(list(cpts_e.values()))
            
        if e:
            prob_e = sum(marginals['p'])
            marginals['p'] = marginals['p'] / prob_e

        return marginals

    def map_and_mpe(self, order_function: Callable, e: List[tuple[str, bool]], M: List[str]=[]) -> tuple[List, pd.DataFrame]:
        """
        Computes the most likely instantiations given evidence and an ordering function (heuristic).
        If M is empty, returns MPE, else it returns MAP

        :param order_function: function for ordering parameters (heuristic)
        :param M: variables in network that we do not want to eliminate or None in case of MPE
        :param e: evidence

        :return: the marginal distribution of MAP_p(M, E) or MPE_p(E)
        """
        Q = self.get_all_variables()
        degrees_occured = []

        if len(M) == 0:
            M = Q
        
        self.net_prune(set(M), e)
        Q = self.get_all_variables()
        
        pi = order_function(list(set(Q) - set(M))) + order_function(M)
        cpts = self.get_all_cpts()
        cpts_e = {}

        for key in cpts:
            cpts_e.update({key: self.reduce_factor(pd.Series(dict(e)), cpts[key])})

        for i in range(len(pi)):
            f_pi, fpi_keys = self._get_cpts_x(pi[i], cpts_e)
            fi = self._mult_factors(f_pi)
            if pi[i] in M:
                fi = self._max_out(fi, pi[i], set(pi[:i]))
            else:
                fi = self._sum_out(fi, pi[i])
            if fi.empty:
                degrees_occured.append(0)
                cpts_e = self._replace_cpts(cpts_e, fpi_keys, {})
                continue


            degrees_occured.append(len(set(fi.columns.values)-set(['p','instantiations'])))
            cpts_e = self._replace_cpts(cpts_e, fpi_keys, {pi[i]: fi})

        return degrees_occured, self._mult_factors(list(cpts_e.values()))

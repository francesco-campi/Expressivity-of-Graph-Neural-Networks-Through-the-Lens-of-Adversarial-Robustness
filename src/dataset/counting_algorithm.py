# For now we only cont size 3 subgraphs

# REMARK: The subsgraphs types should have a fixed order 
import networkx as nx
import numpy as np
from scipy.special import comb


def subgraph_counting_all(graph: nx.Graph):
    """Counts the the subgraphs fro all the types we consider and sores teh result in a dictionary. 
    The types can be in {'gij', '3-Star not ind.'}, where 'gij' represents the graphlet with i nodes of type j according 
    to our list of graphlets, see paper from Nino Shervashidze for reference

    :param graph: _description_
    :type graph: nx.Graph
    :return: Returns a dictionary where the key is the type of the graphlets and value is count if all the graphlet have been conted,
    :rtype: _type_
    """
    # initialize the counts
    counts = {'Triangle':0, '2-Path':0, '4-Clique': 0, 'Chordal cycle': 0, 'Tailed triangle': 0, '3-Star': 0, '4-Cycle': 0, '3-Path': 0, '3-Star not ind.': 0}
    # store the nodes and the neighbours for better performances
    nodes = set(graph.nodes)
    neighbors = {}
    for v in nodes:
        neighbors[v] = set(graph.neighbors(v))
    
    # counts the subsgraphs
    for v1 in nodes:
        ######### star-like subgraph #######
        counts['3-Star not ind.'] += int(comb(len(neighbors[v1]), 3))
        for v2 in filter(lambda v: v > v1, neighbors[v1]):

            ######## subgraphs of size 3 #########
            # subgraph Triangle
            counts['Triangle'] += len(neighbors[v1] & neighbors[v2])
            # subgraph 2-Path
            counts['2-Path'] += len(neighbors[v1] - (neighbors[v2] | {v2}))
            counts['2-Path'] += len(neighbors[v2] - (neighbors[v1] | {v1}))

            ######## subgraphs of size 4 ##########
            # case 1: update the counts of the motifs containg a triangle
            for v3 in neighbors[v1] & neighbors[v2]:
                # subsgraph 4-Clique
                counts['4-Clique'] += len(neighbors[v1] & neighbors[v2] & neighbors[v3])
                # subsgraph Chordal cycle
                counts['Chordal cycle'] += len((nodes - (neighbors[v1] | {v1})) & neighbors[v2] & neighbors[v3])
                counts['Chordal cycle'] += len((nodes - (neighbors[v2] | {v2})) & neighbors[v1] & neighbors[v3])
                counts['Chordal cycle'] += len((nodes - (neighbors[v3] | {v3})) & neighbors[v1] & neighbors[v2])
                # subsgraph Tailed triangle
                counts['Tailed triangle'] += len(neighbors[v1] & (nodes - (neighbors[v2] | neighbors[v3] | {v2, v3}))) # not in every case v2, v3 are neighbours but we want to be sure not to counts them
                counts['Tailed triangle'] += len(neighbors[v2] & (nodes - (neighbors[v1] | neighbors[v3] | {v1, v3})))
                counts['Tailed triangle'] += len(neighbors[v3] & (nodes - (neighbors[v1] | neighbors[v2] | {v1, v2})))
            # case 2
            for v3 in neighbors[v1] - (neighbors[v2] | {v2}):
                # subsgraph Chordal cycle
                counts['Chordal cycle'] += len(neighbors[v1] & neighbors[v2] & neighbors[v3]) # already counted for k
                # subsgraph 4-Cycle
                counts['4-Cycle'] += len((nodes - (neighbors[v1] | {v1})) & neighbors[v2] & neighbors[v3])
                # subsgraph Tailed triangle
                counts['Tailed triangle'] += len((nodes - (neighbors[v2] | {v2})) & neighbors[v1] & neighbors[v3])
                counts['Tailed triangle'] += len((nodes - (neighbors[v3] | {v3})) & neighbors[v1] & neighbors[v2])
                # subsgraph 3-Star
                counts['3-Star'] += len(neighbors[v1] & (nodes - (neighbors[v2] | neighbors[v3] | {v2, v3})))
                # subsgraph 3-Path
                counts['3-Path'] += len(neighbors[v2] & (nodes - (neighbors[v1] | neighbors[v3] | {v1, v3})))
                counts['3-Path'] += len(neighbors[v3] & (nodes - (neighbors[v1] | neighbors[v2] | {v1, v2})))
            # case 3
            for v3 in neighbors[v2] - (neighbors[v1] | {v1}):
                # subsgraph Chordal cycle
                counts['Chordal cycle'] += len(neighbors[v1] & neighbors[v2] & neighbors[v3]) # already counted for k
                # subsgraph 4-Cycle
                counts['4-Cycle'] += len((nodes - (neighbors[v2] | {v2})) & neighbors[v1] & neighbors[v3]) #already counted for k
                # subsgraph Tailed triangle
                counts['Tailed triangle'] += len((nodes - (neighbors[v1] | {v1})) & neighbors[v2] & neighbors[v3])
                counts['Tailed triangle'] += len((nodes - (neighbors[v3] | {v3})) & neighbors[v1] & neighbors[v2])
                # subsgraph 3-Star
                counts['3-Star'] += len(neighbors[v2] & (nodes - (neighbors[v1] | neighbors[v3] | {v1, v3})))
                # subsgraph 3-Path
                counts['3-Path'] += len(neighbors[v1] & (nodes - (neighbors[v2] | neighbors[v3] | {v2, v3})))
                counts['3-Path'] += len(neighbors[v3] & (nodes - (neighbors[v1] | neighbors[v2] | {v1, v2})))
    
    # normalize the counts
    counts['Triangle'] = int(counts['Triangle'] / 3)
    counts['2-Path'] = int(counts['2-Path'] / 2)
    counts['4-Clique'] = int(counts['4-Clique'] / 12) #24
    counts['Chordal cycle'] = int(counts['Chordal cycle'] / 10) #20
    counts['Tailed triangle'] = int(counts['Tailed triangle'] / 7) #16
    counts['3-Star'] = int(counts['3-Star'] / 6) #12
    counts['4-Cycle'] = int(counts['4-Cycle'] / 8) #16
    counts['3-Path'] = int(counts['3-Path'] / 4) #12
    return counts


def subgraph_counting(graph: nx.Graph, subgraph_type: str = 'Triangle') -> int:
    """Counting the subsgraphs of the given graphs. The motivs that are counted are of the type given in input.
    The types can be in {'gij', '3-Star not ind.'}, where 'gij' represents the graphlet with i nodes of type j according 
    to our list of graphlets, see paper from Nino Shervashidze for reference

    :param graph: graph on which perform the counting
    :type graph: networkx.Graph
    :param subraphs_type: type of subraphs we want to counts
    :type subraphs_type: str
    :return: Returns the count of the given substructure
    :rtype: int
    """


    def count_Triangle(graph):
        # initialize the counts
        count = 0
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # counts the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]): #considers each edge only once
                #substrucures with 3 edge
                count += len(neighbors[v1] & neighbors[v2])
        # normalize the counts
        count = int(count/3)
        return count
    
    def count_2_Path(graph):
        # initialize the counts
        count = 0
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # counts the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]):
                # subgraph 2-Path
                count += len(neighbors[v1] - (neighbors[v2] | {v2}))
                count += len(neighbors[v2] - (neighbors[v1] | {v1}))
        # normalize the counts
        count = int(count / 2)
        return count


    def count_4_Clique(graph):
        # initialize the counts
        count = 0
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # counts the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]):
                # case 1: update the counts of the motifs containg a triangle
                for v3 in neighbors[v1] & neighbors[v2]:
                    # subsgraph 4-Clique
                    count += len(neighbors[v1] & neighbors[v2] & neighbors[v3])
        
        # normalize the counts
        count = int(count / 12) #24
        return count

    def count_Chordal_cycle(graph):
        # initialize the counts
        count = 0
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # counts the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]):
                # case 1: update the counts of the motifs containg a triangle
                for v3 in neighbors[v1] & neighbors[v2]:
                    # subsgraph Chordal cycle
                    count += len((nodes - (neighbors[v1] | {v1})) & neighbors[v2] & neighbors[v3])
                    count += len((nodes - (neighbors[v2] | {v2})) & neighbors[v1] & neighbors[v3])
                    count += len((nodes - (neighbors[v3] | {v3})) & neighbors[v1] & neighbors[v2])
                # case 2
                for v3 in neighbors[v1] - (neighbors[v2] | {v2}):
                    # subsgraph Chordal cycle
                    count += len(neighbors[v1] & neighbors[v2] & neighbors[v3]) # already counted for k
                # case 3
                for v3 in neighbors[v2] - (neighbors[v1] | {v1}):
                    # subsgraph Chordal cycle
                    count += len(neighbors[v1] & neighbors[v2] & neighbors[v3]) # already counted for k
        
        # normalize the counts
        count = int(count / 10) #20
        return count


    def count_Tailed_triangle(graph):
        # initialize the counts
        count = 0
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # counts the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]):
                # case 1: update the counts of the motifs containg a triangle
                for v3 in neighbors[v1] & neighbors[v2]:
                    # subsgraph Tailed triangle
                    count += len(neighbors[v1] & (nodes - (neighbors[v2] | neighbors[v3] | {v2, v3}))) # not in every case v2, v3 are neighbours but we want to be sure not to counts them
                    count += len(neighbors[v2] & (nodes - (neighbors[v1] | neighbors[v3] | {v1, v3})))
                    count += len(neighbors[v3] & (nodes - (neighbors[v1] | neighbors[v2] | {v1, v2})))
                # case 2
                for v3 in neighbors[v1] - (neighbors[v2] | {v2}):
                    # subsgraph Tailed triangle
                    count += len((nodes - (neighbors[v2] | {v2})) & neighbors[v1] & neighbors[v3])
                    count += len((nodes - (neighbors[v3] | {v3})) & neighbors[v1] & neighbors[v2])
                # case 3
                for v3 in neighbors[v2] - (neighbors[v1] | {v1}):
                    # subsgraph Tailed triangle
                    count += len((nodes - (neighbors[v1] | {v1})) & neighbors[v2] & neighbors[v3])
                    count += len((nodes - (neighbors[v3] | {v3})) & neighbors[v1] & neighbors[v2])
        
        # normalize the counts
        count = int(count / 7) #16
        return count
    
    def count_Tailed_triangle_not_ind(graph):
        A = nx.to_numpy_array(graph)
        A2=A.dot(A)
        A3=A2.dot(A)
        count=((np.diag(A3)/2)*(A.sum(0)-2)).sum()
        return count

    def count_3_Star(graph):
        # initialize the counts
        count = 0
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # counts the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]):
                # case 2
                for v3 in neighbors[v1] - (neighbors[v2] | {v2}):
                    # subsgraph 3-Star
                    count += len(neighbors[v1] & (nodes - (neighbors[v2] | neighbors[v3] | {v2, v3})))
                # case 3
                for v3 in neighbors[v2] - (neighbors[v1] | {v1}):
                    # subsgraph 3-Star
                    count += len(neighbors[v2] & (nodes - (neighbors[v1] | neighbors[v3] | {v1, v3})))
        
        # normalize the counts
        count = int(count / 6) #12
        return count

    def count_4_Cycle(graph):
        # initialize the counts
        count = 0
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # counts the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]):
                # case 2
                for v3 in neighbors[v1] - (neighbors[v2] | {v2}):
                    # subsgraph 4-Cycle
                    count += len((nodes - (neighbors[v1] | {v1})) & neighbors[v2] & neighbors[v3])
                # case 3
                for v3 in neighbors[v2] - (neighbors[v1] | {v1}):
                    # subsgraph 4-Cycle
                    count += len((nodes - (neighbors[v2] | {v2})) & neighbors[v1] & neighbors[v3]) #already counted for k
        
        # normalize the counts
        count = int(count / 8) #16
        return count

    def count_4_Cycle_not_ind(graph):
        A = nx.to_numpy_array(graph)
        A2=A.dot(A)
        A3=A2.dot(A)
        count=1/8*(np.trace(A3.dot(A))+np.trace(A2)-2*A2.sum())
        return count

    def count_3_Path(graph):
        # initialize the counts
        count = 0
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # counts the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]):
                # case 2
                for v3 in neighbors[v1] - (neighbors[v2] | {v2}):
                    # subsgraph 3-Path
                    count += len(neighbors[v2] & (nodes - (neighbors[v1] | neighbors[v3] | {v1, v3})))
                    count += len(neighbors[v3] & (nodes - (neighbors[v1] | neighbors[v2] | {v1, v2})))
                # case 3
                for v3 in neighbors[v2] - (neighbors[v1] | {v1}):
                    # subsgraph 3-Path
                    count += len(neighbors[v1] & (nodes - (neighbors[v2] | neighbors[v3] | {v2, v3})))
                    count += len(neighbors[v3] & (nodes - (neighbors[v1] | neighbors[v2] | {v1, v2})))
        
        # normalize the counts
        count = int(count / 4) #12
        return count


    def count_3_Star_not_ind(graph):
        """Count induced and not induced star-shaped structures (1 center and 3 rays)
        """
        count = 0
        for v in graph.nodes:
            count += int(comb(len(list(graph.neighbors(v))), 3))
        return count


    # check the sizes given in input are correct
    if subgraph_type == 'Triangle':
        count = count_Triangle(graph)
    elif subgraph_type == '2-Path':
        count = count_2_Path(graph)
    elif subgraph_type == '4-Clique':
        count = count_4_Clique(graph)
    elif subgraph_type == 'Chordal cycle':
        count = count_Chordal_cycle(graph)
    elif subgraph_type == 'Tailed triangle':
        count = count_Tailed_triangle(graph)
    elif subgraph_type == '3-Star':
        count = count_3_Star(graph)
    elif subgraph_type == '4-Cycle':
        count = count_4_Cycle(graph)
    elif subgraph_type == '3-Path':
        count = count_3_Path(graph)
    elif subgraph_type == '3-Star not ind.':
        count = count_3_Star_not_ind(graph)
    elif subgraph_type == 'Tailed triangle not ind.':
        count = count_Tailed_triangle_not_ind(graph)
    elif subgraph_type == '4-Cycle not ind.':
        count = count_4_Cycle_not_ind(graph)
    # Raise an error if the size is incorrect
    else:
        raise ValueError(f"The subgraph type {subgraph_type} is not supported!")

    return count


def subgraph_listing_all(graph:nx.Graph):
    """Lists the the subgraphs fro all the types we consider and sores the result in a dictionary. 

    :param graph: _description_
    :type graph: nx.Graph
    :return: Returns a dictionary where the key is the type of the graphlets and value is count if all the graphlet have been conted,
    :rtype: _type_
    """
    # initialize the counts
    subgraphs = {'Triangle':set(), '2-Path':set(), '4-Clique': set(), 'Chordal cycle': set(), 'Tailed triangle': set(), '3-Star': set(), '4-Cycle': set(), '3-Path': set(),}# '3-Star not ind.': set()}
    # store the nodes and the neighbours for better performances
    nodes = set(graph.nodes)
    neighbors = {}
    for v in nodes:
        neighbors[v] = set(graph.neighbors(v))
    
    # counts the subsgraphs
    for v1 in nodes:
        ######### star-like subgraph #######
        # counts['3-Star not ind.'] += int(comb(len(neighbors[v1]), 3)) TODO
        for v2 in filter(lambda v: v > v1, neighbors[v1]):

            ######## subgraphs of size 3 #########
            # subgraph Triangle
            v3_all = neighbors[v1] & neighbors[v2]
            for v3 in v3_all:
                subgraphs['Triangle'].add(frozenset([v1, v2, v3]))
            # subgraph 2-Path
            v3_all = (neighbors[v1] - (neighbors[v2] | {v2}))
            v3_all |= (neighbors[v2] - (neighbors[v1] | {v1}))
            for v3 in v3_all:
                subgraphs['2-Path'].add(frozenset([v1, v2, v3]))

            ######## subgraphs of size 4 ##########
            # case 1: update the counts of the motifs containg a triangle
            for v3 in neighbors[v1] & neighbors[v2]:
                # subsgraph 4-Clique
                v4_all = (neighbors[v1] & neighbors[v2] & neighbors[v3])
                for v4 in v4_all:
                    subgraphs['4-Clique'].add(frozenset([v1, v2, v3, v4]))
                # subsgraph Chordal cycle
                v4_all = ((nodes - (neighbors[v1] | {v1})) & neighbors[v2] & neighbors[v3])
                v4_all |= ((nodes - (neighbors[v2] | {v2})) & neighbors[v1] & neighbors[v3])
                v4_all |= ((nodes - (neighbors[v3] | {v3})) & neighbors[v1] & neighbors[v2])
                for v4 in v4_all:
                    subgraphs['Chordal cycle'].add(frozenset([v1, v2, v3, v4]))
                # subsgraph Tailed triangle
                v4_all = (neighbors[v1] & (nodes - (neighbors[v2] | neighbors[v3] | {v2, v3}))) # not in every case v2, v3 are neighbours but we want to be sure not to counts them
                v4_all |= (neighbors[v2] & (nodes - (neighbors[v1] | neighbors[v3] | {v1, v3})))
                v4_all |= (neighbors[v3] & (nodes - (neighbors[v1] | neighbors[v2] | {v1, v2})))
                for v4 in v4_all:
                    subgraphs['Tailed triangle'].add(frozenset([v1, v2, v3, v4]))
            # case 2
            for v3 in neighbors[v1] - (neighbors[v2] | {v2}):
                # subsgraph Chordal cycle
                v4_all = (neighbors[v1] & neighbors[v2] & neighbors[v3]) # already counted for k
                for v4 in v4_all:
                    subgraphs['Chordal cycle'].add(frozenset([v1, v2, v3, v4]))
                # subsgraph 4-Cycle
                v4_all = ((nodes - (neighbors[v1] | {v1})) & neighbors[v2] & neighbors[v3])
                for v4 in v4_all:
                    subgraphs['4-Cycle'].add(frozenset([v1, v2, v3, v4]))
                # subsgraph Tailed triangle
                v4_all = ((nodes - (neighbors[v2] | {v2})) & neighbors[v1] & neighbors[v3])
                v4_all |= ((nodes - (neighbors[v3] | {v3})) & neighbors[v1] & neighbors[v2])
                for v4 in v4_all:
                    subgraphs['Tailed triangle'].add(frozenset([v1, v2, v3, v4]))
                # subsgraph 3-Star
                v4_all = (neighbors[v1] & (nodes - (neighbors[v2] | neighbors[v3] | {v2, v3})))
                for v4 in v4_all:
                    subgraphs['3-Star'].add(frozenset([v1, v2, v3, v4]))
                # subsgraph 3-Path
                v4_all = (neighbors[v2] & (nodes - (neighbors[v1] | neighbors[v3] | {v1, v3})))
                v4_all |= (neighbors[v3] & (nodes - (neighbors[v1] | neighbors[v2] | {v1, v2})))
                for v4 in v4_all:
                    subgraphs['3-Path'].add(frozenset([v1, v2, v3, v4]))
            # case 3
            for v3 in neighbors[v2] - (neighbors[v1] | {v1}):
                # subsgraph Chordal cycle
                v4_all = (neighbors[v1] & neighbors[v2] & neighbors[v3])
                for v4 in v4_all:
                    subgraphs['Chordal cycle'].add(frozenset([v1, v2, v3, v4]))
                # subsgraph 4-Cycle
                v4_all = ((nodes - (neighbors[v2] | {v2})) & neighbors[v1] & neighbors[v3])
                for v4 in v4_all:
                    subgraphs['4-Cycle'].add(frozenset([v1, v2, v3, v4]))
                # subsgraph Tailed triangle
                v4_all = ((nodes - (neighbors[v1] | {v1})) & neighbors[v2] & neighbors[v3])
                v4_all |= ((nodes - (neighbors[v3] | {v3})) & neighbors[v1] & neighbors[v2])
                for v4 in v4_all:
                    subgraphs['Tailed triangle'].add(frozenset([v1, v2, v3, v4]))
                # subsgraph 3-Star
                v4_all = (neighbors[v2] & (nodes - (neighbors[v1] | neighbors[v3] | {v1, v3})))
                for v4 in v4_all:
                    subgraphs['3-Star'].add(frozenset([v1, v2, v3, v4]))
                # subsgraph 3-Path
                v4_all = (neighbors[v1] & (nodes - (neighbors[v2] | neighbors[v3] | {v2, v3})))
                v4_all |= (neighbors[v3] & (nodes - (neighbors[v1] | neighbors[v2] | {v1, v2})))
                for v4 in v4_all:
                    subgraphs['3-Path'].add(frozenset([v1, v2, v3, v4]))
    
    return subgraphs


def subgraph_listing(graph: nx.Graph, subgraph_type: str = 'Triangle') -> set:
    """Listing the subsgraphs of the given graphs. The motivs that are counted are of the type given in input.

    :param graph: graph on which perform the counting
    :type graph: networkx.Graph
    :param subraphs_type: type of subraphs we want to counts
    :type subraphs_type: str
    :return: Returns the count of the given substructure
    :rtype: int
    """


    def list_Triangle(graph):
        # initialize the subgraphs
        subgraphs = set()
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # list  the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]): #considers each edge only once
                #substrucures with 3 edge
                v3_all = neighbors[v1] & neighbors[v2]
                for v3 in v3_all:
                    subgraphs.add(frozenset([v1, v2, v3]))
        return subgraphs
    
    def list_2_Path(graph):
        # initialize the subgraphs
        subgraphs = set()
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # list the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]):
                # subgraph 2-Path
                v3_all = (neighbors[v1] - (neighbors[v2] | {v2})) 
                v3_all |= (neighbors[v2] - (neighbors[v1] | {v1}))
                for v3 in v3_all:
                    subgraphs.add(frozenset([v1, v2, v3]))
        return subgraphs


    def list_4_Clique(graph):
        # initialize the subgraphs
        subgraphs = set()
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # list the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]):
                # case 1: update the counts of the motifs containg a triangle
                for v3 in neighbors[v1] & neighbors[v2]:
                    # subsgraph 4-Clique
                    v4_all = neighbors[v1] & neighbors[v2] & neighbors[v3]
                    for v4 in v4_all:
                        subgraphs.add(frozenset([v1, v2, v3, v4]))
        return subgraphs

    def list_Chordal_cycle(graph):
        # initialize the subgraphs
        subgraphs = set()
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # list the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]):
                # case 1: update the counts of the motifs containg a triangle
                for v3 in neighbors[v1] & neighbors[v2]:
                    # subsgraph Chordal cycle
                    v4_all = ((nodes - (neighbors[v1] | {v1})) & neighbors[v2] & neighbors[v3]) 
                    v4_all |= ((nodes - (neighbors[v2] | {v2})) & neighbors[v1] & neighbors[v3])
                    v4_all |= ((nodes - (neighbors[v3] | {v3})) & neighbors[v1] & neighbors[v2])
                    for v4 in v4_all:
                        subgraphs.add(frozenset([v1, v2, v3, v4]))
                # case 2
                for v3 in neighbors[v1] - (neighbors[v2] | {v2}):
                    # subsgraph Chordal cycle
                    v4_all = (neighbors[v1] & neighbors[v2] & neighbors[v3])
                    for v4 in v4_all:
                        subgraphs.add(frozenset([v1, v2, v3, v4]))
                # case 3
                for v3 in neighbors[v2] - (neighbors[v1] | {v1}):
                    # subsgraph Chordal cycle
                    v4_all = (neighbors[v1] & neighbors[v2] & neighbors[v3])
                    for v4 in v4_all:
                        subgraphs.add(frozenset([v1, v2, v3, v4]))
        
        return subgraphs


    def list_Tailed_triangle(graph):
        # initialize the subgraphs
        subgraphs = set()
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # list the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]):
                # case 1: update the counts of the motifs containg a triangle
                for v3 in neighbors[v1] & neighbors[v2]:
                    # subsgraph Tailed triangle
                    v4_all = (neighbors[v1] & (nodes - (neighbors[v2] | neighbors[v3] | {v2, v3}))) # not in every case v2, v3 are neighbours but we want to be sure not to counts them
                    v4_all |= (neighbors[v2] & (nodes - (neighbors[v1] | neighbors[v3] | {v1, v3})))
                    v4_all |= (neighbors[v3] & (nodes - (neighbors[v1] | neighbors[v2] | {v1, v2})))
                    for v4 in v4_all:
                        subgraphs.add(frozenset([v1, v2, v3, v4]))
                # case 2
                for v3 in neighbors[v1] - (neighbors[v2] | {v2}):
                    # subsgraph Tailed triangle
                    v4_all= ((nodes - (neighbors[v2] | {v2})) & neighbors[v1] & neighbors[v3])
                    v4_all |= ((nodes - (neighbors[v3] | {v3})) & neighbors[v1] & neighbors[v2])
                    for v4 in v4_all:
                        subgraphs.add(frozenset([v1, v2, v3, v4]))
                # case 3
                for v3 in neighbors[v2] - (neighbors[v1] | {v1}):
                    # subsgraph Tailed triangle
                    v4_all = ((nodes - (neighbors[v1] | {v1})) & neighbors[v2] & neighbors[v3])
                    v4_all |= ((nodes - (neighbors[v3] | {v3})) & neighbors[v1] & neighbors[v2])
                    for v4 in v4_all:
                        subgraphs.add(frozenset([v1, v2, v3, v4]))
        
        return subgraphs
    
    # def count_Tailed_triangle_not_ind(graph):
    #     A = nx.to_numpy_array(graph)
    #     A2=A.dot(A)
    #     A3=A2.dot(A)
    #     count=((np.diag(A3)/2)*(A.sum(0)-2)).sum()
    #     return count

    def list_3_Star(graph):
        # initialize the subgraphs
        subgraphs = set()
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # list the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]):
                # case 2
                for v3 in neighbors[v1] - (neighbors[v2] | {v2}):
                    # subsgraph 3-Star
                    v4_all = (neighbors[v1] & (nodes - (neighbors[v2] | neighbors[v3] | {v2, v3})))
                    for v4 in v4_all:
                        subgraphs.add(frozenset([v1, v2, v3, v4]))
                # case 3
                for v3 in neighbors[v2] - (neighbors[v1] | {v1}):
                    # subsgraph 3-Star
                    v4_all = (neighbors[v2] & (nodes - (neighbors[v1] | neighbors[v3] | {v1, v3})))
                    for v4 in v4_all:
                        subgraphs.add(frozenset([v1, v2, v3, v4]))
        
        return subgraphs

    def list_4_Cycle(graph):
        # initialize the subgraphs
        subgraphs = set()
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # list the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]):
                # case 2
                for v3 in neighbors[v1] - (neighbors[v2] | {v2}):
                    # subsgraph 4-Cycle
                    v4_all = ((nodes - (neighbors[v1] | {v1})) & neighbors[v2] & neighbors[v3])
                    for v4 in v4_all:
                        subgraphs.add(frozenset([v1, v2, v3, v4]))
                # case 3
                for v3 in neighbors[v2] - (neighbors[v1] | {v1}):
                    # subsgraph 4-Cycle
                    v4_all = ((nodes - (neighbors[v2] | {v2})) & neighbors[v1] & neighbors[v3]) 
                    for v4 in v4_all:
                        subgraphs.add(frozenset([v1, v2, v3, v4]))
        
        return subgraphs

    # def count_4_Cycle_not_ind(graph):
    #     A = nx.to_numpy_array(graph)
    #     A2=A.dot(A)
    #     A3=A2.dot(A)
    #     count=1/8*(np.trace(A3.dot(A))+np.trace(A2)-2*A2.sum())
    #     return count

    def list_3_Path(graph):
        # initialize the subgraphs
        subgraphs = set()
        # store the nodes and the neighbours for better performances
        nodes = set(graph.nodes)
        neighbors = {}
        for v in nodes:
            neighbors[v] = set(graph.neighbors(v))
        # list the subsgraphs
        for v1 in nodes:
            for v2 in filter(lambda v: v > v1, neighbors[v1]):
                # case 2
                for v3 in neighbors[v1] - (neighbors[v2] | {v2}):
                    # subsgraph 3-Path
                    v4_all = (neighbors[v2] & (nodes - (neighbors[v1] | neighbors[v3] | {v1, v3})))
                    v4_all |= (neighbors[v3] & (nodes - (neighbors[v1] | neighbors[v2] | {v1, v2})))
                    for v4 in v4_all:
                        subgraphs.add(frozenset([v1, v2, v3, v4]))
                # case 3
                for v3 in neighbors[v2] - (neighbors[v1] | {v1}):
                    # subsgraph 3-Path
                    v4_all = (neighbors[v1] & (nodes - (neighbors[v2] | neighbors[v3] | {v2, v3})))
                    v4_all |= (neighbors[v3] & (nodes - (neighbors[v1] | neighbors[v2] | {v1, v2})))
                    for v4 in v4_all:
                        subgraphs.add(frozenset([v1, v2, v3, v4]))
        
        return subgraphs


    # def count_3_Star_not_ind(graph):
    #     """Count induced and not induced star-shaped structures (1 center and 3 rays)
    #     """
    #     count = 0
    #     for v in graph.nodes:
    #         count += int(comb(len(list(graph.neighbors(v))), 3))
    #     return count


    # check the sizes given in input are correct
    if subgraph_type == 'Triangle':
        subgraphs = list_Triangle(graph)
    elif subgraph_type == '2-Path':
        subgraphs = list_2_Path(graph)
    elif subgraph_type == '4-Clique':
        subgraphs = list_4_Clique(graph)
    elif subgraph_type == 'Chordal cycle':
        subgraphs = list_Chordal_cycle(graph)
    elif subgraph_type == 'Tailed triangle':
        subgraphs = list_Tailed_triangle(graph)
    elif subgraph_type == '3-Star':
        subgraphs = list_3_Star(graph)
    elif subgraph_type == '4-Cycle':
        subgraphs = list_4_Cycle(graph)
    elif subgraph_type == '3-Path':
        subgraphs = list_3_Path(graph)
    # elif subgraph_type == '3-Star not ind.':
    #     subgraphs = list_3_Star_not_ind(graph)
    # elif subgraph_type == 'Tailed triangle not ind.':
    #     subgraphs = list_Tailed_triangle_not_ind(graph)
    # elif subgraph_type == '4-Cycle not ind.':
    #     subgraphs = list_4_Cycle_not_ind(graph)
    # Raise an error if the size is incorrect
    else:
        raise ValueError(f"The subgraph type {subgraph_type} is not supported!")

    return subgraphs
def sparsify_additive_generator(g, fracEdges):
    '''
    Implements the sparsification algorithm described in 
    
    Bryan Wilder and Gita Sukthankar. "Sparsification of Social Networks Using Random Walks". SocialCom 2015.
    
    g is a NetworkX graph to be sparsified (assumed undirected). fracEdges is a list containing the desired
    levels of sparsification, specified by the fraction of the total number of edges which should be included.
    For instance, fracEdges = [0.01, 0.05, 0.1] would return networks with 1%, 5%, and 10% of the total edges.
    
    Function is written as a generator, so it successively yields each graph. 
    '''
    import networkx as nx
    import heapq
    import math
    
    points = [] #stores the objective value at each iteration
    edge_heap = [] #min-heap which stores the edges for greedy to add
    entry_lookup = {} #maps from edges to entries in the heap: u,v
    
    edgePoints = [int(g.number_of_edges()*i) for i in fracEdges]
    
    if type(g) is nx.DiGraph:
        gsparse = nx.DiGraph()
    else:
        gsparse = nx.Graph()
    gsparse.add_nodes_from(g.nodes())
    
    #probability of jumping to a random node
    tau = .05
    #compute the constant which should be added to get the total weight in the graph with those edges
    c = (1./g.number_of_nodes()) * (tau / (1.-tau)) * g.number_of_edges()
    totalweight =  2*g.number_of_edges() + g.number_of_edges() * (tau/(1-tau))
    #holds the time in the stationary distribution for each node
    pi = {}
    for i in g.nodes():
        pi[i] = (g.degree(i) + c)/(totalweight)
    #compute the marginal gain in KL divergence for adding each edge
    #each entry in the heap has [-marginal gain, node 1, node 1, valid (boolean)]
    #python's minheap provides the element with least negative marginal gain
    for u,v in g.edges():
        if not gsparse.has_edge(u, v):
            weight = pi[u] * math.log((gsparse.degree(u) + 1 + c)/(gsparse.degree(u) + c)) + pi[v] * math.log((gsparse.degree(v) + 1 + c)/(gsparse.degree(v) + c))
            weight *= -1
            entry = [weight, u, v, True]        
            edge_heap.append(entry)
            entry_lookup[(u, v)] = entry
            
    heapq.heapify(edge_heap)
    iterations = 0
    totaledges = len(g.edges())
    #maintain a list of edges whose marginal gain evaluation is invalid (since
    #an edge was added to one of their endpoints)
    invalid = []
    #iteratively add edges
    while iterations < int(fracEdges[-1] * totaledges):
        iterations += 1
        print(iterations)
        #rebuild the heap if the top element is not valid
        if not edge_heap[0][3]:
            heapq.heapify(edge_heap)
            for e in invalid:
                e[3] = True
            invalid = []
        #pop the top element off the heap and add it to the sparse graph
        removed = heapq.heappop(edge_heap)
        points.append(-removed[0])
        gsparse.add_edge(removed[1], removed[2])
        #need to update edges iff they share a vertex with the edge just added
        for i in (removed[1], removed[2]):
            for u,v in g.edges(i):
                if not gsparse.has_edge(u, v):
                    #compute new weight and modify data structures
                    weight = pi[u] * math.log((gsparse.degree(u) + 1 + c)/(gsparse.degree(u) + c)) + pi[v] * math.log((gsparse.degree(v) + 1 + c)/(gsparse.degree(v) + c))
                    weight *= -1 
                    if entry_lookup.has_key((u, v)):
                        k = (u,v)
                    else:
                        k = (v,u)
                    entry_lookup[k][0] = weight
                    entry_lookup[k][3] = False
                    #mark its heap position as invalid (could actually be lower)
                    invalid.append(entry_lookup[k])
        #yield the current network if asked
        if iterations in edgePoints:
            yield gsparse.copy()
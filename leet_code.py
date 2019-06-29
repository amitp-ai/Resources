#Graph Search Algorithms: DFS, BFS, Dijkstra/UCS, A_Star, and Dynamic Programming
print('Graph Search Algorithms: DFS, BFS, Dijkstra/UCS, A_Star, and Dynamic Programming')
def find_path(path, strt, stp):
    #Used for DFS, BFS, Dijkstra, A_star, DP_ver1, DP_ver2
    res = []
    n = stp
    while n != strt:
        res.append(n)
        n = path[n]
    res.append(n)
    res.reverse()
    return res


def DFS(G):
    #Used to find path in an unqeighted Graph
    class Node():
        def __init__(self,node,running_cost,parent):
            self.node = node
            self.running_cost = running_cost
            self.parent = parent
        def __str__(self):
            return str(self.node) + ' ' + str(self.running_cost)

    #find path length from 'a' to 'h'
    start = 'a'
    goal = 'h'
    stack = []
    nodes_visited = {}
    path = {}
    stack.append(Node(start, 0, None))
    while stack != []:
        #print([str(n) for n in stack])
        node = stack.pop() #visit a node
        if node.node in nodes_visited: #This is only need to properly trace the path (otherwise it's not necessary)
            continue #multiple instance of the same node.node could be in the queue. But as soon as a single one of them is visited, the rest are useless and should be ignored.
        nodes_visited[node.node] = True #mark it as visited so we don't visit it again
        path[node.node] = node.parent
        if node.node == goal:
            print(find_path(path, start, goal))
            return node.running_cost
        node_children = G[node.node]
        for c in node_children:
            if c not in nodes_visited:
                stack.append(Node(c,node.running_cost+1,node.node))
    return None

def BFS(G):
    #Used to find shortest path in an unweighted Graph
    class Node():
        def __init__(self,node,running_cost,parent):
            self.node = node
            self.running_cost = running_cost
            self.parent = parent
        def __str__(self):
            return str(self.node) + ' ' + str(self.running_cost)

    #find path length from 'a' to 'h'
    start = 'a'
    goal = 'h'
    queue = []
    nodes_visited = {}
    path = {}
    queue.append(Node(start, 0, None))
    while queue != []:
        #print([str(n) for n in queue])
        node = queue.pop(0) #visit a node
        if node.node in nodes_visited: #This is only need to properly trace the path (otherwise it's not necessary)
            continue #multiple instance of the same node.node could be in the queue. But as soon as a single one of them is visited, the rest are useless and should be ignored.
        nodes_visited[node.node] = True #mark it as visited so we don't visit it again
        path[node.node] = node.parent
        if node.node == goal:
            print(find_path(path, start, goal))
            return node.running_cost
        node_children = G[node.node]
        for c in node_children:
            if c not in nodes_visited:
                queue.append(Node(c,node.running_cost+1,node.node))
    return None


import heapq
def Dijkstra_UCS(G):
    #Used to find shortest path in a weighted Graph (Dijkstra is same as Uniform Cost Search)
    #It is like BFS but the edges are weighted. So need to use a special type of queue, i.e. priority queue
    #Requirement: All the weights must be non-negative otherwise Dijkstra doesn't work
    class Node(object):
        def __init__(self,node,running_cost, parent):
            self.node = node
            self.running_cost = running_cost
            self.parent = parent

        def __lt__(self, other_node):
            if self.running_cost < other_node.running_cost:
                return True
            else:
                return False
        
        def __str__(self):
            return str(self.node) + ' ' + str(self.running_cost)

    #Find path length from 'a' to 'h'
    start = 'a'
    goal = 'h'
    pqueue = []
    heapq.heapify(pqueue)
    nodes_visited = {}
    path = {}
    heapq.heappush(pqueue,Node(start,0,None))
    while pqueue != []:
        #print([str(n) for n in pqueue])
        node = heapq.heappop(pqueue) #visit a node according to some priority
        if node.node in nodes_visited: #This is only need to properly trace the path (otherwise it's not necessary)
            continue #multiple instance of the same node.node could be in the queue. But as soon as a single one of them is visited, the rest are useless and should be ignored.
        nodes_visited[node.node] = True #mark the node as visited so as not to re-visit it again
        path[node.node] = node.parent
        if node.node == goal:
            print(find_path(path,start,goal))
            return node.running_cost
        node_children = G[node.node]
        for c in node_children.keys():
            if c not in nodes_visited:
                heapq.heappush(pqueue, Node(c,node.running_cost+node_children[c],node.node))
    return None



def A_star(G):
    #Used to find shorted path in a weighted Graph
    #It is more efficient than Dijkstra's
    #Requirement: All the weights must be non-negative otherwise Dijkstra doesn't work
    #Find path from 'a' to 'h'

    #Use Manhattan distance (i.e. L2 norm) for cost to go
    node_numeric_val = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g': 7, 'h':8, 'z1': 9, 'z2': 10}


    class Node(object):
        def __init__(self,node,running_cost,parent):
            self.node = node
            self.running_cost = running_cost
            self.cost_to_goal = (node_numeric_val[node] - node_numeric_val['h']) ** 2
            self.parent = parent
            self.a_star_cost = self.cost_to_goal + self.running_cost

        def __lt__(self, other_node):
            if self.a_star_cost < other_node.a_star_cost:
                return True
            else:
                return False

        def __str__(self):
            return str(self.node) + ' ' + str(self.running_cost)

    #Find path length from 'a' to 'h'
    start = 'a'
    goal = 'h'
    pqueue = []
    heapq.heapify(pqueue)
    nodes_visited = {}
    path = {}
    heapq.heappush(pqueue, Node(start,0,None))
    while pqueue != []:
        #print([str(n) for n in pqueue])
        node = heapq.heappop(pqueue)
        if node.node in nodes_visited: #This is only need to properly trace the path (otherwise it's not necessary)
            continue #multiple instance of the same node.node could be in the queue. But as soon as a single one of them is visited, the rest are useless and should be ignored.
        nodes_visited[node.node] = True
        path[node.node] = node.parent
        if node.node == goal:
            print(find_path(path, start, goal))
            return node.running_cost
        node_children = G[node.node]
        for c in node_children.keys():
            if c not in nodes_visited:
                new_node = Node(c, node.running_cost+node_children[c], node.node)
                heapq.heappush(pqueue, new_node)
    return None


def Dynamic_Programming_ver1(G):
    #Used to find shorted path in a Graph (from every node to the goal)
    #Verison 1 approach is based upon value iteration (and so it is slightly less efficient than version 2 but it is easy to code)

    #Find path from 'a' to 'h'
    start = 'a'
    goal = 'h'

    #states are all the nodes
    state_values = {}
    best_action_at_state = {} #to trace the path
    for k in G.keys():
        state_values[k] = 999 #some large number for fast convergence (in RL use small number for exploration, but b'cse we have the model, no need to explore)
    state_values[goal] = 0

    num_iter = 10
    for i in range(num_iter):
        for k in G.keys():
            if k == goal:
                continue
            node_children = G[k]
            min_val = 999
            best_action = None
            for c in node_children.keys():
                if node_children[c] + state_values[c] < min_val:
                    best_action = c
                    min_val = node_children[c] + state_values[c]
            state_values[k] = min_val
            best_action_at_state[k] = best_action
        #print(best_action_at_state)
    path = find_path(best_action_at_state, goal, start)
    path.reverse()
    print(path)



def Dynamic_Programming_ver2(G):
    #Used to find shorted path in a Graph (from every node to the goal)
    #Version 2 is more efficient where we find the path tracing back from the goal
        #However, since the graph is represented using outgoing edges from every not an not incoming edges to each node, it is not computationally cheap to backtrack from the goal node
        #and minimize the 'cost to go.'
        #However, we can instead minimize the 'running cost from the start node.' And this will still give us the same result, and it will be compuitationally cheaper to implement.

        start = 'a'
        goal = 'h'
        state_values = {}
        best_action_at_state = {} #to trace the path
        for n in G.keys():
            state_values[n] = 999 #initialize to some large number for fast convergence
            best_action_at_state[n] = None       
        state_values[start] = 0
        nodes_to_try_queue = [start]
        while nodes_to_try_queue != []:
            #print(nodes_to_try_queue)
            node = nodes_to_try_queue.pop(0)
            if node == goal:
                break
            node_children = G[node]
            for c in node_children.keys():
                if (node_children[c] + state_values[node]) < state_values[c]:
                    state_values[c] = (node_children[c] + state_values[node])
                    best_action_at_state[c] = node
                nodes_to_try_queue.append(c)
        print(best_action_at_state)
        print(find_path(best_action_at_state, start, goal))


#Represent Graph using adjacency list (this Graph represent's outgoing edges for each node, instead of incoming edges)
Graph_adj_list = {'a': ['b','c'], 'b': ['a','d'], 'c': ['a','g'], 'd':['e','c','z1'],
      'e':['f'], 'f':[], 'g':['d','h'], 'h':[], 'z1':['z2'], 'z2':['h']}

#Represent Graph using adjacency map (this Graph represent's outgoing edges for each node, instead of incoming edges)
Graph_adj_map_weighted = {'a': {'b':1,'c':1}, 'b': {'a':1,'d':2}, 'c': {'a':1,'g':1}, 'd':{'e':4,'c':2,'z1':1},
      'e':{'f':1}, 'f':{}, 'g':{'d':3,'h':5}, 'h':{}, 'z1':{'z2':3}, 'z2':{'h':6}}



print('A. GRAPH SEARCH')
print('1: DFS')
print(DFS(Graph_adj_list))
print('2: BFS')
print(BFS(Graph_adj_list))
print('3: Dijkstra/UCS')
print(Dijkstra_UCS(Graph_adj_map_weighted))
print('4: A_Star')
print(A_star(Graph_adj_map_weighted))
print('5: Dynamic Programming - Value Iteration Based')
print(Dynamic_Programming_ver1(Graph_adj_map_weighted))
print('6: Dynamic Programming - Staring from Goal Approach')
print(Dynamic_Programming_ver2(Graph_adj_map_weighted))

print()
#do DP and Dijkstra for this matrix
print('B. MATRIX PATH TRAVERSAL (application of Graph Search)')
matrix = [[0,2,3,4],
          [5,6,7,8],
          [9,10,11,12],
          [13,14,15,16]]

matrix = [[0,10,10,10],
          [1,20,20,20],
          [100,1,1,0]]


import heapq
def Dijkstra_UCS_Matrix(matrix):
    #find shortest path from start node to goal node
    #allowable actions are go right and go down
    #Dijkstra only works if the edges are non-negative
    class Node():
        def __init__(self, idx, running_cost, parent):
            self.idx = idx
            self.running_cost = running_cost
            self.parent = parent

        def __lt__(self, other_node):
            if self.running_cost < other_node.running_cost:
                return True
            else:
                return False

    r,c = len(matrix)-1, len(matrix[0])-1
    start, goal = (0,0), (r,c)
    pqueue = []
    heapq.heapify(pqueue)
    node = Node(start,0, None)
    heapq.heappush(pqueue,node)
    nodes_visited = {}
    path = {}
    while pqueue != []:
        node = heapq.heappop(pqueue)
        if node.idx in nodes_visited: #This is only need to properly trace the path (otherwise it's not necessary)
            continue #multiple instance of the same node.idx could be in the queue. But as soon as a single one of them is visited, the rest are useless and should be ignored.
        nodes_visited[node.idx] = True
        path[node.idx] = node.parent
        #print(nodes_visited)
        #print(path)
        #print()
        if node.idx == goal:
            print(find_path(path, start, goal))
            return node.running_cost

        #add unvisited children of node to the pqueue
        i,j = node.idx
        if (i+1 <= r) and ((i+1,j) not in nodes_visited):
            new_node = Node((i+1,j), node.running_cost+matrix[i+1][j], node.idx)
            heapq.heappush(pqueue, new_node)
        if (j+1 <= c) and ((i,j+1) not in nodes_visited):
            new_node = Node((i,j+1), node.running_cost+matrix[i][j+1], node.idx)
            heapq.heappush(pqueue, new_node)
    return None



def Dynamic_Programming_ver2_matrix(matrix):
    #find shortest path from start node to goal node
    #allowable actions are go right and go down
    #DP works even if the edges are negative

    #the states are each of the posiitons in the matrix
    r,c = len(matrix)-1, len(matrix[0])-1
    start, goal = (0,0), (r,c)
    state_values = {}
    best_action_at_state = {}
    for i in range(r+1):
        for j in range(c+1):
            state_values[(i,j)] = 999 #initialize to some large value
    state_values[goal] = 0

    #so we start from the goal and minimize the cost to goal
    nodes_to_expand_queue = [(r,c)] #to keep track of which nodes to expand
    while nodes_to_expand_queue != []:
        node = nodes_to_expand_queue.pop(0)
        if node == start:
            #print(state_values)
            #print(best_action_at_state)
            path = find_path(best_action_at_state, goal, start)
            path.reverse()
            print(path)
            print(state_values[start])
            break
        i,j = node

        #Find the children
        #actions allowed are left and up(as we are backtracking to find which prev_node led us to node)
        if i-1 >= 0:
            nodes_to_expand_queue.append((i-1,j))
            temp_val = matrix[i][j] + state_values[(i,j)]
            if temp_val < state_values[(i-1,j)]:
                state_values[(i-1,j)] = temp_val
                best_action_at_state[(i-1,j)] = (i,j) #'down'
        if j-1 >= 0:
            nodes_to_expand_queue.append((i,j-1))
            temp_val = matrix[i][j] + state_values[(i,j)]
            if temp_val < state_values[(i,j-1)]:
                state_values[(i,j-1)] = temp_val
                best_action_at_state[(i,j-1)] = (i,j) #'right'



print('1. Dijkstra/UCS')
print(Dijkstra_UCS_Matrix(matrix))
print('2. Dynamic Programming ver 2')
print(Dynamic_Programming_ver2_matrix(matrix))

print()
print('C. MAX SUM SUB ARRAY')
def max_sum_subarray_Kadane(array):
    curr_sum = array[0]
    max_sum = array[0]
    for i in range(1,len(array)):
        curr_sum = max(curr_sum+array[i], array[i])
        max_sum = max(max_sum, curr_sum)
    return max_sum


import heapq
def max_sum_subarray_UCS(array):
    #Tree search based approach using Dijkstra/Uniform Cost Search
    class Node():
        def __init__(self, idxs, sub_sum):
            self.idxs = idxs #this needs to be a list if in additona to the max_sub_sum, we want the max_sub_sum_array
            self.sub_sum = sub_sum
        def __lt__(self, other_node):
            if self.sub_sum > other_node.sub_sum:
                return True
            else:
                return False

    pqueue = []
    heapq.heapify(pqueue)
    for idx in range(len(array)):
        heapq.heappush(pqueue, Node([idx], array[idx]))
    max_sum = -999
    max_sum_subarray = None
    nodes_visited = {}
    while pqueue != []:
        #print([(n.idxs, n.sub_sum) for n in pqueue])
        node = heapq.heappop(pqueue)
        nodes_visited[node] = True
        if node.sub_sum > max_sum:
            max_sum = node.sub_sum
            max_sum_subarray = node.idxs

        #add children to the queue (i.e. expand right)
        #print(node.idxs)
        pos = node.idxs[-1]
        if (pos + 1) < len(array):
            new_node = Node(node.idxs+[pos+1], node.sub_sum+array[pos+1])
            if new_node not in nodes_visited: #and new_node.sub_sum > max_sum: #to speed up, only add a child if its better than max_sum (No it doesn't work!!!)
                heapq.heappush(pqueue, new_node)
    return [max_sum, max_sum_subarray]



array = [1,2,3,-10,1,5,9,-11,2,1]
print('1. Using Kadanes Algorith O(N)')
print(max_sum_subarray_Kadane(array))
print('2. Dijkstra/UCS O(N^2)')
print(max_sum_subarray_UCS(array))


def dp_robbery_bad(array):
    n = len(array)
    state_values = {}
    for i in range(n):
        state_values[i] = 0
    state_values[n-1] = array[n-1]

    num_iters = 10
    for _ in range(num_iters):
        for i in range(n-1):

            #find next actions

            #circular array            
            next_action_states = []
            j = (i+2)%n
            if i == 0: k = n-1
            else: k = i-1
            while True:
                if j == k or j == i:
                    break
                next_action_states.append(j)
                j = (j+1)%n


            #non-circular array (only can go right)
            next_action_states = range(i+2,n)

            max_next_state = 0
            #print(i, next_action_states)
            for ns in next_action_states:
                max_next_state = max(max_next_state, state_values[ns])
            state_values[i] = array[i] + max_next_state
        print(state_values)


def dp_robbery_ver1(array):
    #non cyclical case (going left to right)
    #Goal: starting from pos=0, find the maximum possible rewards I could get

    value_state = {}
    value_state[0] = array[0]
    value_state[1] = max(array[0], array[1])

    for idx in range(2, len(array)):
        value_state[idx] = max(value_state[idx-2]+array[idx], value_state[idx-1])
    print(value_state)


def dp_robbery_ver2(array):
    #non cyclical case (going left to right)
    #Goal: starting from pos=0, find the maximum possible rewards I could get
    #value iteration based

    value_state = {}
    for idx in range(len(array)):
        value_state[idx] = -999
    #anchor the last two state values (as they can't change in non-cyclical case)
    value_state[len(array)-1] = array[len(array)-1]
    value_state[len(array)-2] = max(array[len(array)-1], array[len(array)-2])

    num_iters = 10
    for _ in range(num_iters):
        for idx in range(0, len(array)-2):
            value_state[idx] = max(value_state[idx], value_state[idx+2]+array[idx], value_state[idx+1])
        print(value_state)


import heapq
def Dijkstra_robbery(array):
    def helper(array, i):
        class Node():
            def __init__(self, idx, run_sum):
                self.idx = idx
                self.run_sum = run_sum

            def __lt__(self, other_node):
                if self.run_sum > other_node.run_sum:
                    return True
                else:
                    return False

            def __str__(self):
                return str(self.idx) + ' ' + str(self.run_sum)


        n = len(array)
        pqueue = []
        heapq.heapify(pqueue)
        new_node = Node(i, array[i])
        heapq.heappush(pqueue, new_node)
        nodes_visited = {}
        max_rewards = 0
        while pqueue != []:
            node = heapq.heappop(pqueue)
            #nodes_visited[node.idx] = True #no don't do this
            max_rewards = max(max_rewards, node.run_sum)

            #add next states

            #circular array
            next_action_states = []
            j = (node.idx+2)%n
            if i == 0: k = n-1
            else: k = i-1
            while True:
                if j == k or j == i:
                    break
                next_action_states.append(j)
                j = (j+1)%n


            #non-circular array (only can go right)
            #next_action_states = range(node.idx+2,n)

            for ns in next_action_states:
                if ns not in nodes_visited:
                    new_node = Node(ns, array[ns]+node.run_sum)
                    heapq.heappush(pqueue, new_node)

        return max_rewards

    state_values = {}
    for i in range(len(array)):
        state_values[i] = helper(array, i)
    print(state_values)



def rob_leetcode_soln(array):
    val_nm1 = 0
    val_nm2 = 0
    #state_value = {}
    for idx,n in enumerate(array):
        temp = val_nm1
        val_nm1 = max(val_nm2+n, val_nm1)
        val_nm2 = temp
        #state_value[idx] = val_nm1
    #print(state_value)
    print(val_nm1)



array = [1,2,3,1000,1,2,1000,9]
#       [0,1,2,3,   4,5,6,   7]
array = [1,2,3,10,-10,4,1,100,-200]
print('ROBBERY VER1')
dp_robbery_ver1(array)
print('ROBBERY VER2')
dp_robbery_ver2(array)
print('ROBBERY BAD')
dp_robbery_bad(array)
print('ROBBERY DIJKSTRA')
Dijkstra_robbery(array)
print('ROBBERY LEET CODE SOLN')
rob_leetcode_soln(array)

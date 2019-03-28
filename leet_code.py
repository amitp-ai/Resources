#test code



'''
Find the shorted path from start node to the target node using a few different algorithms:
1. Using simple DFS with multiple iterations to find the optimal solution
2. Using Uniform Cost Search (Dijkstra's algorithm). This is basically an extension of Breadth First Search
3. Using A* algorithm (Basically sort the priority queue using the sum of true distance from start node to current node and optimistic sum of current node to target node)
4. Using Dynamic Programming
'''

def shortest_path(array):
    #find the shortest path from top-left to bot-right.
    #can only go right or down (i.e. can't form a circular graph)
    #DFS based implementation. Can also do BFS based implementation by turning the stack into a queue.
    #Keeping track of nodes_visisted is useful when we have cycles in the graph. For trees it is not necessary.
    class TreeNode(object):
        def __init__(self, dist, i, j):
            self.dist = dist
            self.i = i
            self.j = j

    rows = len(array)
    cols = len(array[0])
    i,j = 0,0
    node = TreeNode(array[i][j], i, j)
    stack = []
    stack.append(node)
    nodes_visited = {} #use a hashmap as its faster to search through
    min_sum = 9999
    num_iter = 0
    while stack != []:
        num_iter += 1
        node = stack.pop()
        dist, i, j = node.dist, node.i, node.j
        nodes_visited[i,j] = True

        if i == rows-1 and j == cols-1:
            nodes_visited = {} #use a hashmap as its faster to search through
            if dist < min_sum:
                min_sum = dist

        if i+1 < rows:
            if (i+1,j) not in nodes_visited:
                new_node = TreeNode(dist+array[i+1][j], i+1, j)
                stack.append(new_node)
        if j+1 < cols:
            if (i,j+1) not in nodes_visited:
                new_node = TreeNode(dist+array[i][j+1], i, j+1)
                stack.append(new_node)
    print('Simple DFS: Min_sum is {} and Num_iter is {}'.format(min_sum, num_iter))

      


import heapq
def shortest_path_UCS(array):
    #AKA Dijkstra's Algorithm
    #find the shortest path from top-left to bot-right.
    #can only go right or down (i.e. can't form a circular graph)
    #Using Uniform Cost Search (aka Dijkstra's Algorithm) (i.e. priority queue based)
    #It is basically breadth first search but with non-uniform edges
    #Keeping track of nodes_visisted is useful when we have cycles in the graph. For trees it is not necessary.
    class TreeNode(object):
        def __init__(self, dist, i, j):
            self.dist = dist
            self.i = i
            self.j = j

        def __str__(self):
            return str((self.dist, self.i, self.j))

        def __lt__(self, node_b):
            if self.dist < node_b.dist:
                return True
            else:
                return False


    rows = len(array)
    cols = len(array[0])
    #Initialize distance between the start node and all the other nodes to some very large value (except for the start node itself)
    Dist = {}
    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0:
                Dist[(i,j)] = array[i][j]
            else:
                Dist[(i,j)] = float('inf') #some large number
    i,j = 0,0
    node = TreeNode(Dist[(i,j)], i, j)
    p_queue = []
    heapq.heapify(p_queue)
    heapq.heappush(p_queue, node)
    nodes_visited = {} #use a hashmap as its faster to search through

    num_iter = 0
    while p_queue != []:
        num_iter += 1
        node = heapq.heappop(p_queue)
        i, j = node.i, node.j
        nodes_visited[(i,j)] = True

        if i == rows-1 and j == cols-1:
            break

        #add all the children nodes that are unvisited (and do edge relaxation to find the shortest distance from start to every node in the tree)
        #if there are no cycles in the graph, then no need to check for nodes_visisted or do edge relaxation. This simplies the code.
        if i+1 < rows:
            if (i+1,j) not in nodes_visited:
                Dist[(i+1,j)] = min(Dist[(i+1,j)], Dist[(i,j)]+array[i+1][j]) #edge relaxation
                new_node = TreeNode(Dist[(i+1,j)], i+1, j)
                heapq.heappush(p_queue, new_node)
        if j+1 < cols:
            if (i,j+1) not in nodes_visited:
                Dist[(i,j+1)] = min(Dist[(i,j+1)], Dist[(i,j)]+array[i][j+1]) #edge relaxation
                new_node = TreeNode(Dist[(i,j+1)], i, j+1)
                heapq.heappush(p_queue, new_node)
    print('UCS: Min_sum is {} and Num_iter is {}'.format(Dist[(rows-1,cols-1)], num_iter))


import math
def shortest_path_A_star(array):
    #find the shortest path from top-left to bot-right.
    #can only go right or down (i.e. can't form a circular graph)
    #Using A* algorithm (i.e. priority queue based)
    #It is basically UCS algorithm but with additional term representing the optimal distance to target
    #Keeping track of nodes_visisted is useful when we have cycles in the graph. For trees it is not necessary.
    class TreeNode(object):
        def __init__(self, dist, a_star_sum, i, j):
            self.dist = dist
            self.a_star_sum = a_star_sum
            self.i = i
            self.j = j

        def __str__(self):
            return str((self.dist, self.i, self.j))

        def __lt__(self, node_b):
            if self.a_star_sum < node_b.a_star_sum:
                return True
            else:
                return False

    def l2_distance(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


    rows = len(array)
    cols = len(array[0])
    #Initialize distance between the start node and all the other nodes to some very large value (except for the start node itself)
    Dist = {}
    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0:
                Dist[i,j] = array[i][j]
            else:
                Dist[i,j] = float('inf')
    i,j = 0,0
    node = TreeNode(Dist[i,j], Dist[i,j]+l2_distance((i,j), (rows-1,cols-1)), i, j)
    nodes_visited = {} #use a hash map for faster search
    p_queue = []
    heapq.heapify(p_queue)
    heapq.heappush(p_queue, node)

    num_iter = 0
    while p_queue != []:
        num_iter += 1
        node = heapq.heappop(p_queue)
        i, j = node.i, node.j
        nodes_visited[i,j] = True

        if i == rows-1 and j == cols-1:
            break

        #add all the children nodes that are unvisited (and do edge relaxation to find the shortest distance from start to every node in the tree)
        #if there are no cycles in the graph, then no need to check for nodes_visited or do edge relaxation. This simplies the code.
        if i+1 < rows:
            if (i+1,j) not in nodes_visited:
                Dist[i+1,j] = min(Dist[i+1,j], Dist[i,j]+array[i+1][j])
                new_node = TreeNode(Dist[i+1,j], Dist[i+1,j]+l2_distance((i+1,j),(rows-1,cols-1)), i+1, j)
                heapq.heappush(p_queue, new_node)
        if j+1 < cols:
            if (i,j+1) not in nodes_visited:
                Dist[i,j+1] = min(Dist[i,j+1], Dist[i,j]+array[i][j+1])
                new_node = TreeNode(Dist[i,j+1], Dist[i,j+1]+l2_distance((i,j+1),(rows-1,cols-1)), i, j+1)
                heapq.heappush(p_queue, new_node)
    print('A Star: Min_sum is {} and Num_iter is {}'.format(Dist[rows-1,cols-1], num_iter))


def shortest_path_DP(array):
    #find the shortest path from top-left to bot-right.
    #can only go right or down (i.e. can't form a circular graph)
    #Using Dynamic Programming
    state_values = {}
    rows = len(array)
    cols = len(array[0])   
    num_iter = 0
    i = rows-1
    while i >= 0:
        j = cols-1
        while j >= 0:
            num_iter += 1
            if i == rows-1 and j == cols-1:
                state_values[(i,j)] = array[i][j]
            else:
                q_down, q_right = 9999, 9999 #some large value
                if i+1 < rows: q_down = state_values[(i+1,j)]
                if j+1 < cols: q_right = state_values[(i,j+1)]
                state_values[(i,j)] = array[i][j] + min(q_down, q_right)
            j -= 1
        i -= 1
    print('DP: Min_sum is {} and Num_iter is {}'.format(state_values[(0,0)], num_iter))



array = [[1,12,3,54],[15,6,57,8],[9,100,11,712],[13,14,15,16],[1,2,11,0],[10,2,11,0]]
# array = [[1,2],[3,4]]
shortest_path(array)
shortest_path_UCS(array)
shortest_path_A_star(array)
shortest_path_DP(array)

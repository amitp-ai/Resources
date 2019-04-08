# Test Code


def unique_routes_dfs(rows,cols):
    #start from top left and end at bot right
    #using DP is most efficient
    #can only go right and down

    #using DFS
    node = (0,0)
    stack = []
    stack.append(node)
    num_paths = 0
    while stack != []:
        node = stack.pop()
        i,j = node
        if node == (rows-1,cols-1):
            num_paths += 1
        if i+1 < rows:
            new_node = (i+1,j)
            stack.append(new_node)
        if j+1 < cols:
            new_node = (i,j+1)
            stack.append(new_node)
    return num_paths

def unique_routes_DP(rows,cols):
    #Using DP where the value of each state corresponds to the number of unique paths from that state tot he end state
    #value iteration based approach (still O(N) but little slower)
    #initialize the state values
    values = [[0 for c in range(cols)] for r in range(rows)]
    values[rows-1][cols-1] = 1
    num_iters = 1 #1 is good enough as we are building the state value function from the target
    for _ in range(num_iters):
        for r in range(rows-1,-1,-1): #(rows):
            for c in range(cols-1,-1,-1): #(cols):
                temp_right = 0
                temp_down = 0
                #right
                if c+1 < cols:
                    temp_right = values[r][c+1]
                #down
                if r+1 < rows:
                    temp_down = values[r+1][c]
                if (r,c) != (rows-1,cols-1):
                    values[r][c] = temp_down + temp_right
        #print(values)
    return values[0][0]


rows=5
cols=5
print(unique_routes_dfs(rows,cols))
print(unique_routes_DP(rows,cols))


def wrapper(func, *args):
    def wrapped():
        return func(*args)
    return wrapped

import timeit
dfs_based = wrapper(unique_routes_dfs, rows, cols)
dp_based = wrapper(unique_routes_DP, rows, cols)
print(timeit.timeit(dfs_based, number=100))
print(timeit.timeit(dp_based, number=100))

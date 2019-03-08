#test

def max_sum_path_DP_less_efficient(array):
    #Array elements can be any real value (positive, zero, or negative)
    #Goal is to find the path with maximum sum from top left cell to bottom right cell.
    #Can only go right or down
    #This is basically a Tree search problem

    #Time complexity is O(rows*cols) and space complexity is O(rows*cols)

    #Two actions are available, right and down from each state i.e. cell location. 
    #Except for the bottom-right cell, no cell/state has the option of staying in that cell.
    #i.e. Each cell/state has only two choices, i.e. go right or go down
    rows = len(array)
    cols = len(array[0])

    #Initialize the state value function where each state corresponds to i,j
    V_State = []
    for r in range(rows):
        V_State.append([])
        for c in range(cols):
            V_State[r].append(-100) #some very low value
    V_State[rows-1][cols-1] = array[rows-1][cols-1]

    for _ in range(10):
        for r in range(rows):
            for c in range(cols):
                q_right, q_down = -100, -100 #some very small value by default
                if c+1 <= cols-1:
                    q_right = array[r][c] + V_State[r][c+1]
                if r+1 <= rows-1:
                    q_down = array[r][c] + V_State[r+1][c]
                if r < rows-1 or c < cols-1:
                    V_State[r][c] = max(q_right, q_down)
        print(V_State)

    #Trace the max_sum_path
    r,c = 0,0
    path = [array[r][c]]
    max_sum = array[r][c]
    while True:
        q_right, q_down = -100, -100
        if c < cols-1:
            q_right = array[r][c] + V_State[r][c+1]
        if r < rows-1:
            q_down = array[r][c] + V_State[r+1][c]

        if q_right > q_down:
            c += 1
        else:
            r += 1
        path.append(array[r][c])
        max_sum += array[r][c]
        if r == rows-1 and c == cols-1:
            break
    print(path, max_sum, V_State[0][0])


def max_sum_path_DP_more_efficient(array):
    #Note: a more efficient DP implementation (still O(m*n)) is to start from the end state and walk backward towards the start state.
    #This way only have to iterate once.
    return None



def max_sum_path_DFS(array):
    #Assume all the elements of the array are non-negative
    #Goal is to find the path with maximum sum from top left cell to bottom right cell.
    #Can only go right or down
    #This is basically a Tree search problem
    class Tree_Node():
        def __init__(self, run_sum, r, c):
            self.run_sum = run_sum
            self.r = r
            self.c = c

    #Use DFS
    rows = len(array)
    cols = len(array[0])
    r,c = 0,0 #start state/cell
    new_tree_node = Tree_Node(array[r][c], r, c)
    stack = [new_tree_node]
    max_run_sum = -100 #some small number
    while stack != []:
        node = stack.pop()
        run_sum,r,c = node.run_sum, node.r, node.c
        if run_sum > max_run_sum:
            max_run_sum = run_sum

        if r+1 <= rows-1: #go down
            new_tree_node = Tree_Node(run_sum+array[r+1][c], r+1, c)
            if r+1 < rows-1 or c < cols-1: #don't push the bottom right cell
                stack.append(new_tree_node)
        if c+1 <= cols-1: #go right
            new_tree_node = Tree_Node(run_sum+array[r][c+1], r, c+1)
            if r < rows-1 or c+1 < cols-1: #don't push the bottom right cell
                stack.append(new_tree_node)

    print(max_run_sum)
    return max_run_sum


# array = [[1,-2,3,4],
#          [-5,6,-7,-8],
#          [9,10,-3,0]] #won't work for the DFS implementation

array = [[1,2,3,4],
         [5,6,7,8],
         [9,10,3,0]]
        

max_sum_path_DP_less_efficient(array)
max_sum_path_DFS(array)

#test code



def min_sum_path(matrix):
    #using DFS (can only go right or down)
    class TreeNode():
        def __init__(self, run_sum, r, c):
            self.run_sum = run_sum
            self.r = r
            self.c = c
    rows, cols = len(matrix), len(matrix[0])
    min_sum = 9999 #some large number
    r,c = 0,0
    node = TreeNode(matrix[r][c], r, c)
    stack = []
    stack.append(node)
    while stack != []:
        node = stack.pop()
        r_sum, r, c = node.run_sum, node.r, node.c
        if r == rows-1 and c == cols-1:
            if r_sum < min_sum:
                min_sum = r_sum
        if r+1 < rows:
            new_node = TreeNode(r_sum+matrix[r+1][c], r+1, c)
            stack.append(new_node)                
        if c+1 < cols:
            new_node = TreeNode(r_sum+matrix[r][c+1], r, c+1)
            stack.append(new_node)
    return min_sum

matrix = [[1,2,3],[4,5,6],[7,8,9]]
# matrix = [[1,2],[3,4]]
print(min_sum_path(matrix))

def min_sum_triangle(triangle):
    #can only go downwards and visit adjacent nodes
    class TreeNode():
        def __init__(self, run_sum, i,j):
            self.run_sum = run_sum
            self.i = i
            self.j = j

    if len(triangle) == 0: return None
    i,j = 0,0
    node = TreeNode(triangle[i][j], i, j)
    stack = []
    stack.append(node)
    min_sum = 999 #some large value
    while stack != []:
        node = stack.pop()
        run_sum, i, j = node.run_sum, node.i, node.j

        if i+1 < len(triangle):
            new_node = TreeNode(run_sum+triangle[i+1][j], i+1, j)
            stack.append(new_node)
            new_node = TreeNode(run_sum+triangle[i+1][j+1], i+1, j+1)
            stack.append(new_node)
        else: #reach triangle base
            if run_sum < min_sum:
                min_sum = run_sum
    return min_sum


def min_sum_triangle_DP(triangle):
    #can only go downwards and visit adjacent nodes
    #much more efficient implementation
    state_values = {}
    #start from the bottom
    i = len(triangle)-1
    if i < 0: return None
    for jj in range(len(triangle[i])):
        state_values[(i,jj)] = triangle[i][jj]
    i -= 1
    while i >= 0:
        for jj in range(len(triangle[i])):
            tmp1 = state_values[(i+1,jj)]
            tmp2 = state_values[(i+1, jj+1)]
            state_values[(i,jj)] = triangle[i][jj] + min(tmp1, tmp2)
        i -= 1
    return state_values[(0,0)]



triangle = [[1],[2,3],[4,5,6],[7,8,9,10], [11,12,13,14,15], [11,12,13,14,15,16]]
print(min_sum_triangle(triangle))
print(min_sum_triangle_DP(triangle))

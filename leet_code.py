#leet code testing


#Find the subarray with the maximum sum
def maxSubArraySum(nums):
    #This is Dynamic Programming Based Solution
    in_sz = len(nums)
    if in_sz > 12: return 'Nums has to be lesss than 12 in size'
    alphabet = ['A','B','C','D','E','F','E','G','H','I','J','K'] #Assume max size of nums is 12
    
    #State names
    states = ['strt']
    for state in alphabet[0:in_sz]:
        states.append(state)
    states.append('end')

    #Transition values (i.e. transition rewards)
    r_tr = {}
    for state in states:
        r_tr[state] = {}
    for i in range(0, in_sz):
        state = states[i+1]
        r_tr['strt'][state] = nums[i]
    for i in range(0, in_sz):
        state = states[i+1]
        r_tr[state][states[i+2]] = nums[i+1] if i+1 < in_sz else 0
        r_tr[state]['end'] = 0

    #State values
    V = {}
    #initialize
    init_state_val = -20 #initialize to some very small value to speed up convergence (at the price of exploration, which is fine for DP as we have the state transition model but for RL need to initialize it to a large value for better exploration)
    for s in states:
        V[s] = init_state_val 
    V['end'] = 0

    #print(r_tr)
    #print(V)

    num_iter = 15
    for _ in range(num_iter):
        for s in states[0:-1]:
            max_ = -10
            for a in r_tr[s].keys():
                if r_tr[s][a] + V[a] > max_:
                    max_ = r_tr[s][a] + V[a]
            V[s] = max_
        print(V)

    #Find best path
    print('Best Path')
    s = 'strt'
    while s is not 'end':
        max_ = -10
        print(s)
        for a in r_tr[s].keys():
            if r_tr[s][a] + V[a] > max_:
                max_ = r_tr[s][a] + V[a]
                s_max = a
        s = s_max

nums = [1,2,-10,3,1,4,-6]
#maxSubArraySum(nums)

#Tree Traversal
#Very Good
#https://en.wikipedia.org/wiki/Tree_traversal
class Node(object):
    def __init__(self, val, left_child=None, right_child=None):
        self.val = val
        self.left_child = left_child
        self.right_child = right_child

class Tree(object):
    def __init__(self, root):
        self.root = root

    def preorder_traversal(self, root, tree_nodes_list):
        #preorder recursion based
        #root,left,right
        if root is None: 
            return None
        elif root.left_child is None and root.right_child is None: 
            tree_nodes_list.append(root.val)
        else: 
            tree_nodes_list.append(root.val)
            if root.left_child is not None: 
                self.preorder_traversal(root.left_child, tree_nodes_list)
            if root.right_child is not None: 
                self.preorder_traversal(root.right_child, tree_nodes_list)
        return None


    def inorder_traversal_iter_old(self, root):
        #inorder iterative method
        #left,root,right
        stack = []
        result = []
        node = root
        while True:
            while node != None:
                stack.append(node)
                node = node.left_child

            if stack == []: break

            node = stack.pop()
            result.append(node)
            node = node.right_child

        return result

            
    def preorder_traversal_iter(self, root):
        #preorder iterative method
        #root,left,right
        stack = []
        stack.append(root)
        nodes_visited = []
        while stack != []:
            node = stack.pop()

            ########################################################################
            #The below code is never executed as nodes are not visited more than once by design
            while node in nodes_visited:
                if stack != []: node = stack.pop()
                else: break
            ########################################################################

            nodes_visited.append(node)
            
            #Note: right child is pushed first so that left child is processed first
            if node.right_child is not None:
                stack.append(node.right_child)
            if node.left_child is not None:
                stack.append(node.left_child)

        return nodes_visited


n1 = Node(10)
n2 = Node(5)
n3 = Node(15)
n4 = Node(0)
n5 = Node(6)
n6 = Node(14)
n7 = Node(18)
n8 = Node(17)
n9 = Node(17.5)
n1.left_child = n2
n1.right_child = n3
n2.left_child = n4
n2.right_child = n5
n3.left_child = n6
n3.right_child = n7
#n7.left_child = n8
#n8.right_child = n9

tree_nodes_list = []
myTree = Tree(root=n1)
myTree.preorder_traversal(myTree.root, tree_nodes_list)
print(tree_nodes_list)
print([i.val for i in myTree.preorder_traversal_iter(myTree.root)])

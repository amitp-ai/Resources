# Test Code



class TreeNode(object):
    def __init__(self, data, left=None, right=None):
        self.left = left
        self.right = right
        self.data = data

class BinaryTree(object):
    def __init__(self, root):
        self.root = root
        self.nodes_list = []

    def inorder_traversal(self, node):
        if node != None:
            self.inorder_traversal(node.left)
            self.nodes_list.append(node.data)
            self.inorder_traversal(node.right)

    def node_depth(self, node):
        tmp_node = (self.root, 0)
        stack = []
        stack.append(tmp_node)
        while stack != []:
            tmp_node = stack.pop()
            if tmp_node[0] == node:
                return tmp_node[1]
            if tmp_node[0].left != None:
                stack.append((tmp_node[0].left, tmp_node[1]+1))
            if tmp_node[0].right != None:
                stack.append((tmp_node[0].right, tmp_node[1]+1))
        return None


    def LCA(self, nodea, nodeb):
        #nodea and nodeb are assumed to be different
        def is_in_subtree(root, nodea=nodea, nodeb=nodeb):
            if (root == nodea or root == nodeb):
                return root
            elif (root.left == nodea and root.right == nodeb) or (root.left == nodeb and root.right == nodea):
                return root
            elif root.left != None and root.right != None:
                temp1 = is_in_subtree(root.left, nodea, nodeb)
                temp2 = is_in_subtree(root.right, nodea, nodeb)
                if temp1 != None and temp2 != None:
                    return root
                else:
                    return temp1 or temp2 #whichever is not None or None if both are None
            elif root.left != None:
                return is_in_subtree(root.left, nodea, nodeb)
            elif root.right != None:
                return is_in_subtree(root.right, nodea, nodeb)
            else:
                return None

        return is_in_subtree(self.root, nodea, nodeb)



n1 = TreeNode(1)
n2 = TreeNode(2)
n3 = TreeNode(3)
n4 = TreeNode(4)
n5 = TreeNode(5)
n6 = TreeNode(6)
n7 = TreeNode(7)
n1.left=n2
n1.right=n3
n2.left=n4
n2.right=n5
n3.left=n6
n3.right=n7

bt = BinaryTree(n1)
bt.inorder_traversal(n1)
print(bt.nodes_list)

print(bt.node_depth(n7))

print(bt.LCA(n5,n4).data)




class MinHeap(object):
    #min heap
    def __init__(self):
        self.data = []

    def upheap(self, pos):
        if pos == 1:
            return None
        parent = pos//2
        if self.data[parent-1] > self.data[pos-1]:
            self.data[parent-1], self.data[pos-1] = self.data[pos-1], self.data[parent-1]
            self.upheap(parent)
        return None

    def add(self, num):
        self.data.append(num)
        self.upheap(pos=len(self.data))


    def downheap(self, pos):
        lc_pos = 2*pos
        rc_pos = 2*pos+1

        if rc_pos <= len(self.data):
            if self.data[lc_pos-1] < self.data[rc_pos-1] and self.data[lc_pos-1] < self.data[pos-1]:
                self.data[pos-1], self.data[lc_pos-1] = self.data[lc_pos-1], self.data[pos-1]
                pos = lc_pos
                self.downheap(pos)
            elif self.data[rc_pos-1] < self.data[lc_pos-1] and self.data[rc_pos-1] < self.data[pos-1]:
                self.data[pos-1], self.data[rc_pos-1] = self.data[rc_pos-1], self.data[pos-1]
                pos = rc_pos
                self.downheap(pos)
        elif lc_pos <= len(self.data):
            if self.data[lc_pos-1] < self.data[pos-1]:
                self.data[pos-1], self.data[lc_pos-1] = self.data[lc_pos-1], self.data[pos-1]
                pos = lc_pos
                self.downheap(pos)
        return None


    def remove(self):
        min_val = self.data[0]
        temp = self.data.pop()
        if len(self.data) > 1:
            self.data[0] = temp
            self.downheap(pos=1)
        return min_val


myheap = MinHeap()
myheap.add(3)
myheap.add(2)
myheap.add(1)
myheap.add(4)

print(myheap.data)

print(myheap.remove())
print(myheap.data)
print(myheap.remove())
print(myheap.data)
print(myheap.remove())
print(myheap.data)
print(myheap.remove())
print(myheap.data)



#find the median of a stream of numbers
print('\nadd nums:\n')
left, right = [], [] #in reality left and right are implemented using maxheap and minheap, respectively
def add_num(num):
    if len(left) == 0:
        left.append(num)     
    elif len(left) <= len(right):
        if num <= min(right):
            left.append(num)
        else:
            left.append(min(right))
            right.remove(min(right))
            right.append(num)
    else: #len(right) > len(left)
        if num >= max(left):
            right.append(num)
        else:
            right.append(max(left))
            left.remove(max(left))
            left.append(num)
    print(left, right)

def median(left, right):
    if len(left) == len(right):
        return (max(left)+min(right))/2
    else: #left will be longer than right, otherwise
        return max(left)


add_num(3)
add_num(2)
add_num(1)
add_num(4)
add_num(0)
add_num(-1)
print(median(left, right))
add_num(6)
print(median(left, right))


print('find non-repeating num in sorted array in o(logn)')

def find_non_repeat(array):
    n = len(array)
    left = 0
    right = n-1
    pass


def binary_search(array, num):
    n = len(array)
    left = 0
    right = n-1
    while left <= right:
        mid = (left+right)//2
        if array[mid] == num:
            return (mid, num)
        elif array[mid] < num:
            left = mid+1
        elif array[mid] > num:
            right = mid-1
    return None

def binary_search_cont(array, num):
    n = len(array)
    left = 0
    right = n-2
    while left <= right:
        mid = (left+right)//2
        if array[mid] <= num and num <= array[mid+1]:
            return (mid, num)
        elif array[mid] < num:
            left = mid+1
        elif array[mid] > num:
            right = mid-1
    return None


array = [1,2,2,5,5,6,6,11,11,19,19]
#array = [1,2,3,4,5,6,7]
print(binary_search(array, 7))
print(binary_search_cont(array, 7))


#find a single number in a sorted array of double repeating nums except for 1 number
def find_single_num(nums):
    n = len(nums)
    left = 0
    right = n-1
    while left <= right:
        if left == right:
            return (left, nums[left])
        #elif (right-left) == 1: #is not possible given the problem statement
        #    return 'There\'s some issue'

        mid = (left+right)//2
        if nums[mid-1] != nums[mid] and nums[mid+1] != nums[mid]:
            return (mid, nums[mid])
        elif nums[mid-1] != nums[mid]:
            if (mid-left) % 2 != 0:
                right = mid-1
            elif (right-mid-1) % 2 != 0:
                left = mid+1+1
            #else is not possible given the problem statement
        elif nums[mid+1] != nums[mid]:
            if (right-mid) % 2 != 0:
                left = mid+1
            elif (mid-left-1) % 2 != 0:
                right = mid-1-1
            #else is not possible given the problem statement
        #else is not possible given the problem statement


nums = [1,2,2,4,4]
nums = [1,1,2,2,4]
nums = [1,1,2,2,3,3,3.5,4,4,5,5]
print(find_single_num(nums))




#Dijkstra's Algorithm:
print("\nDijkstra's Algorithm")
import heapq

grid = [[0,0,0,0,1],
        [0,1,0,0,0],
        [1,1,1,0,0],
        [0,0,0,1,0],
        [1,1,0,0,0]]

def Dijkstra_Shortest_Path(grid, start=(0,0), goal=(4,4)):
    #Solve using Dijkstra's alogorithm

    #Two arrays: visited and frontier
    #Can take two actions, right or down

    visited = {}
    frontier = []
    heapq.heapify(frontier) #minheap
    heapq.heappush(frontier, (0, start))
    while frontier != []:
        node_cost, node_pos = heapq.heappop(frontier)
        visited[node_pos] = True
        if node_pos == goal:
            return node_cost

        #for right
        new_pos = (node_pos[0], node_pos[1]+1)
        if new_pos[1] < len(grid[0]):
            if grid[new_pos[0]][new_pos[1]] == 0 and new_pos not in visited: #no obstacles and was not previously visited
                new_node = (node_cost+1, new_pos)
                heapq.heappush(frontier, new_node)
        #for down
        new_pos = (node_pos[0]+1, node_pos[1])
        if new_pos[0] < len(grid):
            if grid[new_pos[0]][new_pos[1]] == 0 and new_pos not in visited: #no obstacles and was not previously visited
                new_node = (node_cost+1, new_pos)
                heapq.heappush(frontier, new_node)

    return 'No path'


print(Dijkstra_Shortest_Path(grid))

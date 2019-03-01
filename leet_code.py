#Code testing

class TreeNode():
    def __init__(self, data, left_child=None, right_child=None):
        self.data = data
        self.left_child = left_child
        self.right_child = right_child

    def __str__(self):
        return str(self.data)

class BST():
    def __init__(self, head=None):
        self.head = head

    def add_node(self, data):
        node = self.head
        while True:
            if node == None:
                node = TreeNode(data)
                self.head = node
                break
            if data < node.data:
                if node.left_child != None:
                    node = node.left_child
                else:
                    node.left_child = TreeNode(data)
                    break
            elif data > node.data:
                if node.right_child != None:
                    node = node.right_child
                else:
                    node.right_child = TreeNode(data)
                    break

    def Traverse_level_order(self):
        #level order traversal (uses a queue instead of stack)
        node = self.head
        queue = []
        nodes_visited = []
        queue.append(node)
        while queue != []:
            node = queue.pop(0)
            nodes_visited.append(node)
            if node.left_child != None:
                queue.append(node.left_child)
            if node.right_child != None:
                queue.append(node.right_child)
        print([str(n) for n in nodes_visited])


    def Traverse(self):
        #preorder order traversal (uses a stack)
        node = self.head
        stack = []
        nodes_visited = []
        stack.append(node)
        while stack != []:
            node = stack.pop()
            nodes_visited.append(node)
            if node.right_child != None:
                stack.append(node.right_child)
            
            if node.left_child != None:
                stack.append(node.left_child)
        print([str(n) for n in nodes_visited])


    def find_minimum(self):
        node = self.head
        if node == None:
            return None
        while node != None:
            if node.left_child != None:
                node = node.left_child
            else:
                break
        print(node.data)

    def find_kth_maximum_another_method(self, k=1):
        node = self.head
        if node == None:
            return None
        max_count = 0
        stack = []
        while True:
            if node != None:
                stack.append(node)
                node = node.right_child
            else:
                max_count += 1
                if stack == []: return None
                node = stack.pop()
                if max_count == k:
                    return node
                else:
                    node = node.left_child


    
    def get_num_nodes(self, node):
        #returns the number of nodes in a subtree whose head is 'node'
        if node == None:
            return 0
        return self.get_num_nodes(node.left_child) + 1 + self.get_num_nodes(node.right_child)


    def find_kth_maximum(self, root, k=1):
        if root == None:
            return None

        right_child = root.right_child
        size = self.get_num_nodes(right_child)
        if size + 1 == k:
            return root
        elif size >= k:
            root = root.right_child
            k = k
            return self.find_kth_maximum(root, k)
        else: #i.e. k > size+1
            root = root.left_child
            k = k - (size+1)
            return self.find_kth_maximum(root, k)


    def shortest_tree_depth(self):
        root = self.head
        if root == None:
            return 0

        queue = []
        depth = 1
        queue.append([root,depth])
        while True:
            root, depth = queue.pop(0)
            if root.left_child == None and root.right_child == None:
                return depth
            if root.left_child != None:
                queue.append([root.left_child, depth+1])
            if root.right_child != None:
                queue.append([root.right_child, depth+1])

mybst = BST()
mybst.add_node(10)
mybst.add_node(5)
mybst.add_node(15)
mybst.add_node(2)
mybst.add_node(20)
mybst.add_node(200)
mybst.Traverse()
mybst.find_minimum()
print(mybst.get_num_nodes(mybst.head))
print()
print(mybst.find_kth_maximum(mybst.head, k=3))
print()
print(mybst.shortest_tree_depth())

#test
class Tree_Node():
    def __init__(self, data, left_child=None, right_child=None):
        self.data = data
        self.left_child = left_child
        self.right_child = right_child

    def __str__(self):
        return str(self.data)


class Binary_Tree():
    def __init__(self, root):
        self.root = root
        self.order_list = []
    

    def Traversal_Recursive(self, node):
        if node == None:
            return None
        self.order_list.append(str(node))
        self.Traversal_Recursive(node.left_child)
        self.Traversal_Recursive(node.right_child)


    def Traversal_Iterative(self, node):
        #DFS Traversal (preorder traversal)
        if node == None:
            return None

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



    def Traversal2(self, node_to_check=None):
        #DFS Traversal (preorder traversal)
        stack = []
        edges_visited = {}
        parent = None
        edge = (parent, self.root)
        stack.append(edge)

        while stack != []:
            edge = stack.pop()
            node = edge[1]
            parent = edge[0]
            edges_visited[node] = parent
            if node == node_to_check:
                break

            if node.right_child != None:
                new_edge = (node, node.right_child)
                stack.append(new_edge)
            
            if node.left_child != None:
                new_edge = (node, node.left_child)
                stack.append(new_edge)

        path = [node_to_check.data]
        walk = edges_visited[node_to_check]
        while walk != None:
            path.append(walk.data)
            walk = edges_visited[walk]
        return path




    def Lowest_Common_Ancestor2(self, root, node_a, node_b):
        #similar to preorder traversal but keep track of a node's parents
        if root == None:
            return None

        node_a_path = self.Traversal2(node_a)
        node_b_path = self.Traversal2(node_b)

        # print(node_a_path)
        # print(node_b_path)

        for a in node_a_path:
            for b in node_b_path:
                if a == b:
                    return a
        


    def Lowest_Common_Ancestor(self, root, node_a, node_b):
        #Recursive method
        if root == None:
            return None
        elif root == node_a or root == node_b:
            return root
        else:
            temp_left = self.Lowest_Common_Ancestor(root.left_child, node_a, node_b)
            temp_right = self.Lowest_Common_Ancestor(root.right_child, node_a, node_b)

            if temp_left != None and temp_right != None:
                return root
            elif temp_left != None:
                return temp_left
            elif temp_right != None:
                return temp_right
            else:
                return None



n1 = Tree_Node(10)
n2 = Tree_Node(5)
n3 = Tree_Node(20)
n4 = Tree_Node(2)
n5 = Tree_Node(30)
n6 = Tree_Node(40)
n7 = Tree_Node(-1)
n1.left_child=n2
n1.right_child=n3
n2.left_child=n7
n2.right_child=n4
n3.left_child=n5
n3.right_child=n6

bt = Binary_Tree(n1)
bt.Traversal_Iterative(n1)
bt.Traversal_Recursive(n1)
print(bt.order_list)
print(bt.Lowest_Common_Ancestor2(n1,n4,n6))
print(bt.Lowest_Common_Ancestor(n1,n4,n6))

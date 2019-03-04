#Test code
class Tree_Node():
    def __init__(self, data, left_child=None, right_child=None):
        self.data = data
        self.left_child = left_child
        self.right_child = right_child


class Binary_Tree():
    def __init__(self, root):
        self.root = root


    def Traversal(self):
        #DFS Traversal (preorder traversal)
        stack = []
        nodes_visited = []
        stack.append(self.root)

        while stack != []:
            node = stack.pop()
            nodes_visited.append(node.data)

            if node.right_child != None:
                stack.append(node.right_child)
            
            if node.left_child != None:
                stack.append(node.left_child)
        return nodes_visited


    def Lowest_Common_Ancestor(self, node_a, node_b):
        #similar to preorder traversal but keep track of a node's parents
        class List_Node():
            def __init__(self, tree_node, parent_list_node=None):
                self.tree_node = tree_node
                self.parent_list_node = parent_list_node


        stack = []
        copy_of_stack = []
        stack.append(List_Node(self.root, None))
        copy_of_stack.append(List_Node(self.root, None))
        while stack != []:
            list_node = stack.pop()
            if list_node.tree_node.right_child != None:
                stack.append(List_Node(list_node.tree_node.right_child, list_node))
                copy_of_stack.append(List_Node(list_node.tree_node.right_child, list_node))
            
            if list_node.tree_node.left_child != None:
                stack.append(List_Node(list_node.tree_node.left_child, list_node))
                copy_of_stack.append(List_Node(list_node.tree_node.left_child, list_node))

        for list_node in copy_of_stack:
            if list_node.tree_node == node_a:
                loc_node_a = list_node
            elif list_node.tree_node == node_b:
                loc_node_b = list_node

        print('Node A localization: {}, {}'.format(loc_node_a.tree_node.data, loc_node_a.parent_list_node.tree_node.data))
        print('Node B localization: {}, {}'.format(loc_node_b.tree_node.data, loc_node_b.parent_list_node.tree_node.data))

        walk_a = loc_node_a
        while walk_a != None:
            walk_b = loc_node_b
            while walk_b != None:
                print('walk a {}'.format(walk_a.tree_node.data))
                print('walk b {}'.format(walk_b.tree_node.data))
                if walk_a == walk_b:
                    return walk_a.tree_node.data
                walk_b = walk_b.parent_list_node
            walk_a = walk_a.parent_list_node


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
print(bt.Traversal())
print(bt.Lowest_Common_Ancestor(n5,n6))

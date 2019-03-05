#Test code
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
    

    def Traversal(self, node_to_check=None):
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




    def Lowest_Common_Ancestor(self, root, node_a, node_b):
        #similar to preorder traversal but keep track of a node's parents
        if root == None:
            return None

        node_a_path = self.Traversal(node_a)
        node_b_path = self.Traversal(node_b)

        # print(node_a_path)
        # print(node_b_path)

        for a in node_a_path:
            for b in node_b_path:
                if a == b:
                    return a

#test

def decoder(num_list):
    class Tree_Node():
        def __init__(self, data, pos, left_child=None, right_child=None):
            self.data = data
            self.pos = pos
            self.left_child = left_child
            self.right_child = right_child

    #Alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' #valid number is between 1 and 26 inclusive
    n = len(num_list)
    queue = []
    nodes_visited = []
    start_node = Tree_Node('start', None)
    queue.append(start_node)
    while queue != []:
        node = queue.pop(0)
        nodes_visited.append(node)

        if node.data == 'start':
            pos = 0
            val1 = num_list[pos]
            if val1 != 0:
                new_node = Tree_Node(val1, pos)
                queue.append(new_node)
                node.left_child = new_node #not necessary
                if pos+1 < n:
                    val2 = int(str(num_list[pos]) + str(num_list[pos+1]))
                    if val2 <= 26:
                        new_node = Tree_Node(val2, pos+1)
                        queue.append(new_node)
                        node.right_child = new_node #not necessary
        else:
            pos = node.pos + 1
            if pos == n:
                new_node = Tree_Node('end', pos)
                queue.append(new_node)
                node.left_child = new_node #not necessary                
            if pos < n:
                val1 = num_list[pos]
                if val1 != 0:
                    new_node = Tree_Node(val1, pos)
                    queue.append(new_node)
                    node.left_child = new_node #not necessary
                    if pos+1 < n:
                        val2 = int(str(num_list[pos]) + str(num_list[pos+1]))
                        if val2 <= 26:
                            new_node = Tree_Node(val2, pos+1)
                            queue.append(new_node)
                            node.right_child = new_node #not necessary             

    print([n.data for n in nodes_visited])
    num_decodes = 0
    for n in nodes_visited:
        if n.data == 'end':
            num_decodes += 1
    return num_decodes

num_list = [2,1,2,0,5]
print(decoder(num_list))

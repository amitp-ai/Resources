#test

def boggle_search(board, word):
    #adjacent words are horizontal or vertical

    #first find the location on the board of the first letter in the word

    rows = len(board)
    cols = len(board[0])
    start_locations = []
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == word[0]:
                start_locations.append((r,c))

    if len(start_locations) < 1:
        return False


    #This is Tree search problem (use DFS)
    Stack = [start_locations[0]]
    nodes_visited = []
    word_pos = 0
    while Stack != []:

        #print(Stack)
        #print(nodes_visited)
        #print(word_pos)
        #print()

        node = Stack.pop()
        r, c = node

        if word_pos < len(word) and board[r][c] == word[word_pos]:
            nodes_visited.append(node)
            word_pos += 1
        else:
            nodes_visited.pop()
            word_pos -= 1
            Stack.append(node)
            continue

        if r+1 <= rows-1 and (r+1, c) not in nodes_visited and word_pos < len(word):
            if board[r+1][c] == word[word_pos]:
                Stack.append((r+1,c))
        if r-1 >= 0 and (r-1, c) not in nodes_visited and word_pos < len(word):
            if board[r-1][c] == word[word_pos]:
                Stack.append((r-1,c))
        if c+1 <= cols-1 and (r, c+1) not in nodes_visited and word_pos < len(word):
            if board[r][c+1] == word[word_pos]:
                Stack.append((r,c+1))
        if c-1 >= 0 and (r, c-1) not in nodes_visited and word_pos < len(word):
            if board[r][c-1] == word[word_pos]:
                Stack.append((r,c-1))

    #print(nodes_visited)

    if len(nodes_visited) != len(word):
        return False
    else:
        return True

board = [['A', 'E', 'L'],
         ['E', 'H', 'L'],
         ['L', 'C', 'O']]

word = 'HELLO'
boggle_search(board, word)

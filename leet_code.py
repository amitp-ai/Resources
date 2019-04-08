# Test Code


def find_largest_square(input_matrix):
    def max_square_from_ij(i,j, input_matrix):
        max_sqr_size = min(len(input_matrix)-i, len(input_matrix[0])-j)
        curr_max = 1
        curr_size = curr_max
        break_flag = False
        for sqr_size in range(2,max_sqr_size):
            for right in range(sqr_size-1):
                for down in range(sqr_size-1):
                    if input_matrix[i+right][j+down] == 0:
                        break_flag == True
                    if break_flag: break
                if break_flag: break
            if break_flag:
                curr_size = sqr_size-1
                break
            else:
                curr_size = sqr_size
        if curr_size > curr_max:
            curr_max = curr_size
        return curr_max
    
    max_sqr_size = 1
    for i in range(len(input_matrix)):
        for j in range(len(input_matrix[0])):
            tmp = max_square_from_ij(i,j,input_matrix)
            if tmp > max_sqr_size:
                max_sqr_size = tmp
    return tmp
            
                    
                
input_matrix = [[1,1],[1,1]]
print(find_largest_square(input_matrix))

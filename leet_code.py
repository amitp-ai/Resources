# Test Code


def find_largest_square_slow(input_matrix):
    def max_square_from_ij(i,j, input_matrix):
        max_sqr_size = min(len(input_matrix)-i, len(input_matrix[0])-j)
        curr_max = 0
        curr_size = curr_max
        break_flag = False
        for sqr_size in range(1,max_sqr_size+1):
            for right in range(sqr_size):
                for down in range(sqr_size):
                    if input_matrix[i+right][j+down] == 0:
                        break_flag = True
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
    
    max_sqr_size = 0
    for i in range(len(input_matrix)):
        for j in range(len(input_matrix[0])):
            tmp = max_square_from_ij(i,j,input_matrix)
            if tmp > max_sqr_size:
                max_sqr_size = tmp
    return max_sqr_size**2


def find_largest_square(input_matrix):
    def max_square_from_ij(i,j, input_matrix):
        max_sqr_size = min(len(input_matrix)-i, len(input_matrix[0])-j)
        curr_max = 0
        curr_size = curr_max
        break_flag = False
        for sqr_size in range(1,max_sqr_size+1):
            for down in range(sqr_size):
                if input_matrix[i+down][j+sqr_size-1] == 0:
                    break_flag = True
                if break_flag: break
            if break_flag: break
            
            for right in range(sqr_size):
                if input_matrix[i+sqr_size-1][j+right] == 0:
                    break_flag = True
                if break_flag: break
            if break_flag: break

        if break_flag:
            curr_size = sqr_size-1
        else:
            curr_size = sqr_size
        if curr_size > curr_max:
            curr_max = curr_size
        return curr_max
    
    max_sqr_size = 0
    for i in range(len(input_matrix)):
        for j in range(len(input_matrix[0])):
            tmp = max_square_from_ij(i,j,input_matrix)
            if tmp > max_sqr_size:
                max_sqr_size = tmp
    return max_sqr_size**2
            
def wrapper(func, *args):
    def wrapped():
        return func(args)
    return wrapped

import timeit
input_matrix = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
slow = wrapper(find_largest_square_slow, input_matrix)
fast = wrapper(find_largest_square, input_matrix)
print(timeit.timeit(slow, number=1000))
print(timeit.timeit(fast, number=1000))

print(fast())

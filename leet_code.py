# test
def find_k_minimum(array, k):
    n = len(array)
    if k > n:
        return None

    prev_min = -1000
    for i in range(k):
        min_ = 1000
        for j in range(n):
            if array[j] < min_ and array[j] > prev_min:
                min_ = array[j]
        prev_min = min_
    return prev_min

array = [1,2,0,-10,3,5,6]
print(find_k_minimum(array, 5))

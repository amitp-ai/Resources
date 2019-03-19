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

A : [ 8, 16, 80, 55, 32, 8, 38, 40, 65, 18, 15, 45, 50, 38, 54, 52, 23, 74, 81, 42, 28, 16, 66, 35, 91, 36, 44, 9, 85, 58, 59, 49, 75, 20, 87, 60, 17, 11, 39, 62, 20, 17, 46, 26, 81, 92 ]
B : 9

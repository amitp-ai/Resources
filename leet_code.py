def find_k_minimum(array, k):
    n = len(array)
    if k > n:
        return None

    prev_min = -1000
    counter = 0
    while True:
        min_ = 1000
        for j in range(n):
            if array[j] < min_ and array[j] > prev_min:
                min_ = array[j]
                min_idx = j
            if array[j] == prev_min and j != prev_min_idx:
                k -= 1
        counter += 1
        prev_min = min_
        prev_min_idx = min_idx
        if counter >= k-1:
            break

    return prev_min

A : [ 94, 87, 100, 11, 23, 98, 17, 35, 43, 66, 34, 53, 72, 80, 5, 34, 64, 71, 9, 16, 41, 66, 96 ]
19

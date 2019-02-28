#Code testing


def chocolate_counts(points):
    n = len(points)
    choc_list = [1]*n
    for i in range(1, n):
        if points[i] > points[i-1]:
            choc_list[i] = choc_list[i-1]+1
        elif points[i] == points[i-1]:
            choc_list[i] = choc_list[i-1]
        elif points[i] < points[i-1]:
            choc_list[i] = choc_list[i-1] - 1

    #print(choc_list)

    while True:
        #find min value in choc_list
        min_idx = 0
        for i in range(n):
            if choc_list[i] < choc_list[min_idx]:
                min_idx = i
        if choc_list[min_idx] >= 1: break
        adj = 1 - choc_list[min_idx]
        #check on left side
        i = min_idx
        choc_list[i] += adj
        i = min_idx-1
        while points[i] > points[i+1]:
            if choc_list[i] < choc_list[i+1]:
                choc_list[i] += adj
            i-=1
        #check on right side
        i = min_idx+1
        if i<n:
            while points[i] > points[i-1]:
                if choc_list[i] < choc_list[i-1]:
                    choc_list[i] += adj
                i+=1
                if i >= n: break
    print(choc_list)


points = [1,3,7,2,1,8,9,8]
points = [1,3,7,2,1,8,9,8,7,6,5,4,8]
points = [1,3,7,2,1,8,9,8,7,6,5,4,8,7,6,5,4]
print(points)
chocolate_counts(points)

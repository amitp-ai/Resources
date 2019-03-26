#test code



def Pascal_Triange(n):
    res = []
    if n <= 0: return res
    res.append([1])
    if n == 1: return res
    res.append([1,1])
    if n == 2: return res
    for i in range(2,n):
        temp = []
        temp.append(1)
        for j in range(len(res[i-1])-1):
            temp.append(res[i-1][j] + res[i-1][j+1])
        temp.append(1)
        res.append(temp)
    print(res)
Pascal_Triange(5)

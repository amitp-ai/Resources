# Test Code


def robbery(amounts):
    #DP based solution (start form the ending state)
    values = [0]*(len(amounts)+1)
    values[-2] = amounts[-1]
    values[-1] = 0
    for i in range(len(amounts)-2,-1,-1):
        val_no_rob = values[i+1]
        val_rob = amounts[i] + values[i+2]
        values[i] = max(val_rob, val_no_rob)

    print(values[0:len(amounts)])
    return values[0]


def robbery_ver2(amounts):
    #DP based solution (value iteration based)
    values = [0]*(len(amounts)+1)
    values[-2] = amounts[-1]
    values[-1] = 0
    num_iters = 10
    for _ in range(num_iters):
        for i in range(0,len(amounts)-1):
            val_rob = amounts[i] + values[i+2]
            val_no_rob = values[i+1]
            values[i] = max(val_rob,val_no_rob)
    print(values[0:len(amounts)])
    return values[0]



amounts = [10,20,5,30,50,60]
robbery(amounts)
robbery_ver2(amounts)

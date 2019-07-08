#test
import time
def fib(n):
    #naive Fibonacci
    if n == 0 or n == 1:
        return 1
    return fib(n-1)+fib(n-2)

state_vals = {}
def fib_fast(n):
    #Memoized Fibonacci
    if n == 0 or n == 1:
        return 1

    if (n-1) in state_vals.keys() and (n-2) in state_vals.keys():
        a = state_vals[n-1]
        b = state_vals[n-2]
    elif (n-1) in state_vals.keys():
        a = state_vals[n-1]
        b = fib_fast(n-2)
        state_vals[n-2] = b
    elif (n-2) in state_vals.keys():
        a = fib_fast(n-1)
        state_vals[n-1] = a
        b = state_vals[n-2]
    else:
        a = fib_fast(n-1)
        state_vals[n-1] = a
        b = fib_fast(n-2)
        state_vals[n-2] = b
    return a+b



strt = time.time()
print(fib(10))
print(time.time()-strt)

strt = time.time()
print(fib_fast(200))
print(time.time()-strt)



def robber_leetcode_soln(array):
    val_nm1 = 0
    val_nm2 = 0
    #state_value = {}
    for idx,n in enumerate(array):
        temp = val_nm1
        val_nm1 = max(val_nm2+n, val_nm1)
        val_nm2 = temp
        #state_value[idx] = val_nm1
    #print(state_value)
    print(val_nm1)

def robber_circular(array):
    #do first but not last
    num_2 = array[0]
    num_1 = max(array[1], array[0])
    for i in range(2,len(array)):
        temp = max(array[i]+num_2, num_1)
        if i == len(array)-1:
            temp = num_1
        num_2, num_1 = num_1, temp
    print(num_1)
    max1 = num_1


    #don't do first
    num_2 = array[1]
    num_1 = max(array[2], array[1])
    for i in range(3,len(array)):
        temp = max(array[i]+num_2, num_1)
        num_2, num_1 = num_1, temp
    print(num_1)
    print(max(num_1, max1))



array = [9,2,3,1000,1,2,11]
robber_leetcode_soln(array)
print('circular')
robber_circular(array)


array = [1,2,3,-7,0,3,-9,4]
def max_sum_subarray(array):
    vals = {}
    vals[0]=array[0]
    for i in range(1, len(array)):
        vals[i] = max(array[i]+vals[i-1], array[i])
    print(vals)

max_sum_subarray(array)


array = [0,1,1,0,1,1,1,0]
def max_bool_len(array):
    #bottom up
    n = len(array)
    vals = {}
    vals[n-1] = int(array[n-1] == 1)
    for i in range(n-2,-1,-1):
        if array[i] == 0:
            vals[i] = 0
        else:
            vals[i] = vals[i+1]+1
    print(vals)


max_bool_len(array)



def knap_sack(items, max_weight):
    state_vals = {}
    def helper(w,ln):
        if w <= 0:
            return 0
        if ln <= 0:
            return 0

        if (w,ln) in state_vals:
            return state_vals[(w,ln)]
        else:
            w_str = items[ln-1]['w']
            v_str = items[ln-1]['v']
            if (w-w_str) < 0:
                a = -999
            else:
                a = helper(w-w_str,ln)+v_str
            res = max(a, helper(w,ln-1))
            state_vals[(w,ln)] = res
            return res

    helper(max_weight, len(items))
    print(state_vals[max_weight, len(items)])



items = [{'w':2, 'v':6}, {'w':2, 'v':10}, {'w':3, 'v':12}]
max_weight = 5
knap_sack(items, max_weight)


def coin_change_slow(coins, amount):
    n = len(coins)
    if n == 0 and amount == 0:
        return 0
    elif n == 0:
        return 999
    elif amount < 0:
        return 999
    else:
        pass

    if amount-coins[n-1] < 0:
        res = coin_change(coins[0:n-1], amount)
        return res
    else:
        resa = coin_change(coins, amount-coins[n-1])
        resb = coin_change(coins[0:n-1], amount)
        return min(resa+1, resb)


cache = {}
def coin_change(coins, amount):
    n = len(coins)
    if n == 0 and amount == 0:
        return 0
    elif n == 0:
        return 9999
    elif amount < 0:
        return 9999
    else:
        pass

    if amount-coins[n-1] < 0:
        key = (coins[0:n-1], amount)
        if key in cache:
            res = cache[key]
        else:
            res = coin_change(*key)
            cache[key] = res
        return res
    else:
        keya = (coins, amount-coins[n-1])
        if keya in cache:
            resa = cache[keya]
        else:
            resa = coin_change(*keya)
            cache[keya] = resa

        keyb = (coins[0:n-1], amount)
        if keyb in cache:
            resb = cache[keyb]
        else:
            resb = coin_change(*keyb)
            cache[keyb] = resb
        return min(resa+1, resb)

coins = tuple([181,79,206,169,487,319,262,162,420])
amount = 9999
coins = tuple([120,6,320,300,100,192,212,89,106,461])
amount = 8332
print(coin_change(coins, amount))
print(coin_change_slow(coins, amount))



def target_sum_slow(nums, S):
    n = len(nums)
    def helper(n, target):
        if n == 0 and target == 0:
            return 1
        elif n <= 0:
            return 0

        resa = helper(n-1, target-(nums[n-1]))
        resb = helper(n-1, target-(-nums[n-1]))
        return resa+resb

    return helper(n, S)


def target_sum(nums, S):
    n = len(nums)
    cache = {}
    def helper(n, target):
        if n == 0 and target == 0:
            return 1
        elif n <= 0:
            return 0

        keya = (n-1, target-(nums[n-1]))
        if keya in cache:
            resa = cache[keya]
        else:
            resa = helper(*keya)
            cache[keya] = resa

        keyb = (n-1, target-(-nums[n-1]))
        if keyb in cache:
            resb = cache[keyb]
        else:
            resb = helper(*keyb)
            cache[keyb] = resb
        
        return resa+resb

    return helper(n, S)


array = [1, 1, 1, 1, 1]
target = 3
print(target_sum(array, target))



def maxprofit_slow(prices, fee):
    n = len(prices)
    def helper(idx, mode, p_idx=None):
        if idx > (n-1):
            return 0

        resa = helper(idx+1, mode, p_idx) #keep as is
        if mode == 's':
            resb = (prices[idx]-prices[p_idx]) - fee + helper(idx+1, 'b', None)
        else: #mode == 'b'
            resb = helper(idx+1, 's', idx)
        return max(resa, resb)

    return helper(0, 'b', None)



def maxprofit(prices, fee):
    n = len(prices)
    cache = {}
    def helper(idx, mode, p_idx=None):
        if idx > (n-1):
            return 0

        keya = (idx+1, mode, p_idx)
        if False: #keya in cache:
            resa = cache[keya]
        else:
            resa = helper(*keya) #keep as is
            cache[keya] = resa

        if mode == 's':
            keyb = (idx+1, 'b', None)
            if False: #keyb in cache:
                resb = cache[keyb]
            else:
                resb = (prices[idx]-prices[p_idx]) - fee + helper(*keyb)
                cache[keyb] = resb
        else: #mode == 'b'
            keyb = (idx+1, 's', idx)
            if False: #keyb in cache:
                resb = cache[keyb]
            else:
                resb = helper(*keyb)
                cache[keyb] = resb
        print(keya, resa)
        print(keyb, resb)
        print()
        return max(resa, resb)

    return helper(0, 'b', None)

prices = [1, 3, 2, 8, 4, 9]
fee = 2
print(maxprofit_slow(prices, fee))
print(maxprofit(prices, fee))

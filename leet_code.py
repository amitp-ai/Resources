def twosum(nums, target=0):
    #This runs in O(n) (ignoring the time complexity of sorting, which is O(nlogn))
    #where as the naive approach runs in O(n^2)
    nums = sorted(nums)
    list_res = []
    lo = 0
    hi = len(nums)-1
    while lo < hi:
        summ = nums[lo]+nums[hi]
        if summ < target:
            lo += 1
            while nums[lo-1] == nums[lo]: lo+=1 #get rid of duplicates
        elif summ > target:
            hi -= 1
            while nums[hi+1] == nums[hi]: hi-=1 #get rid of duplicates
        else:
            list_res.append([nums[lo],nums[hi]])
            lo += 1 #or do hi -= 1 but don't do both
            while nums[lo-1] == nums[lo]: lo+=1 #get rid of duplicates
    return list_res

nums = [1,2,-1,6,-5,-2,5,5,-5]
print(twosum(nums))

def threesum(nums, target=0):
    nums = sorted(nums)
    list_res = []

    def twosum(nums, num1, target=0):
        #Assume nums is sorted
        #This runs in O(n)
        #where as the naive approach runs in O(n^2)
        list_res = []
        lo = 0
        hi = len(nums)-1
        while lo < hi:
            summ = nums[lo]+nums[hi]
            if summ < target:
                lo += 1
                while nums[lo-1] == nums[lo]: lo+=1 #get rid of duplicates
            elif summ > target:
                hi -= 1
                while nums[hi+1] == nums[hi]: hi-=1 #get rid of duplicates
            else:
                list_res.append([num1,nums[lo],nums[hi]])
                lo += 1 #or do hi -= 1 but don't do both
                while nums[lo-1] == nums[lo]: lo+=1 #get rid of duplicates
        return list_res

    i=0
    while i <= (len(nums)-2):
        while nums[i]==nums[i+1]: i+=1 #get rid of duplicates
        list_res += twosum(nums[i+1:], nums[i], target=target-nums[i]) #has no duplicates
        #list_res += twosum(nums[0:i]+nums[i+1:], nums[i], target=target-nums[i]) #has duplicates
        i+=1
    return list_res


print(threesum(nums))
print()

def maxSubArray(nums):
    """Find maximum contiguous sub array"""
    summ = nums[0]
    summ_idx = [0]
    max_sum = summ
    max_sum_idx = summ_idx
    for w in range(1,len(nums)):
        val = nums[w]
        if (summ+val) < val:
            summ = val #then forget past history
            summ_idx = [w]        
        else:
            summ += val
            summ_idx.append(w)

        if (summ > max_sum): #only update max_sum if summ is better
            max_sum = summ
            max_sum_idx = tuple(summ_idx)
    return max_sum, max_sum_idx

nums = [-2,1,-3,4,-1,2,1,-5,4]
print(nums)
print(maxSubArray(nums))

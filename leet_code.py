# Test Code
'''
def majority_element(array):
    #Using Boyer-Moore Majority Vote algorithm.
    #Can find results in O(N) time complexity and O(1) space complexity (http://goo.gl/64Nams)
    #Its a 2 steps process: part a and part b

    #1a. find majority > len(array)//2 (there will be at most 1 number)
    count1, candidate1 = 0, 999
    for num in array:
        if num == candidate1:
            count1 += 1
        elif count1 == 0:
            candidate1 = num
            count1 += 1
        else:
            count1 -= 1
    print(candidate1, count1)
    #2b. Then verify if candidate1 really is majority by going through the entire array again

    #2a. find majority > len(array)//3 (there will be at most 2 numbers)
    count1, candidate1 = 0, 999
    count2, candidate2 = 0, 999
    for num in array:
        if num == candidate1:
            count1 += 1
        elif num == candidate2:
            count2 += 1
        elif count1 == 0:
            candidate1 = num
            count1 += 1
        elif count2 == 0:
            candidate2 = num
            count2 += 1
        else:
            count1 -= 1
            count2 -= 1
    print(candidate1, count1, candidate2, count2)
     #2b. Then verify if candidate1/candidate2 really are majority by going through the entire array again


array = [1,2,3,4,5,6,7,1]
array = [1,2,1,1,1,3,4,2,1,2,1,1,2,2,6]
array = [1,2,1,1,1,3,4,2,1,2,1,1,2,-2,6]
array = [1,2,3]
majority_element(array)

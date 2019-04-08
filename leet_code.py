# Test Code

def quadruple_sum(integer_list, target):
    #integer_list is sorted
    #runs in O(N^3)
    res = []
    for i in range(len(integer_list)-3):
        for j in range(i+1, len(integer_list)-2):
            doublesum_target = target-integer_list[i]-integer_list[j]
            doublesum_list = integer_list[j+1:]
            left, right = 0, len(doublesum_list)-1
            while left < right:
                if doublesum_list[left] + doublesum_list[right] == doublesum_target:
                    res.append([integer_list[i], integer_list[j], doublesum_list[left], doublesum_list[right]])
                    left += 1
                    right -= 1
                elif doublesum_list[left] + doublesum_list[right] < doublesum_target:
                    left += 1
                else: #doublesum_list[left] + doublesum_list[right] > doublesum_target:
                    right -= 1
    return res

integer_list = [0,1,2,3,4,5,6,7,8,9,10]
target = 10
print(quadruple_sum(integer_list, target))

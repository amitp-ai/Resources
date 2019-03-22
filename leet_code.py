#test code

print('Test code')

#Search Algorithms
'''
1. if the array is not sorted, then it take O(n) time using Linear Search to search for an element
2. If the array is sorted, then it take O(nlogn) using Binary Search to search for an element
3. Using a hash table (i.e. map e.g. Python dictionary), it take O(1) to search for an element (as the hash function will directly find the location of the element in the hash table/array if it exists)
'''

#Sort Algorithms
def selection_sort(array):
    #Time complexity: O(n) and Space complexity: O(1)
    n = len(array)
    for i in range(n-1):
        min_ = array[i]
        for j in range(i+1,n):
            if array[j] < min_: min_ = array[j]
        array[i] = min_
    return array

def insertion_sort(array):
    #Time complexity: O(n) and Space complexity: O(1)
    n = len(array)
    i = 1
    while i < n:
        j = i
        while array[j] < array[j-1] and j > 0:
            temp = array[j]
            array[j] = array[j-1]
            array[j-1] = temp
            j -= 1
        i += 1
    return array

def bubble_sort(array):
    #Time complexity: O(n) and Space complexity: O(1)
    n = len(array)
    for i in range(n):
        for j in range(1,n-i): #2x more efficient than 'for j in range(1,n):'
            if array[j] < array[j-1]:
                temp = array[j-1]
                array[j-1] = array[j]
                array[j] = temp
    return array

def merge_sort(array):
    #Time complexity: O(nlogn) and Space complexity: O(n)
    #Very difficult to make merge sort be in place (i.e. space complexity of O(1)). But can do so with quick-sort and heap-sort
    def merge(a1, a2):
        n1 = len(a1)
        n2 = len(a2)
        a = []
        i = 0
        j = 0
        while i < n1 or j < n2:
            if i < n1 and j < n2:
                if a1[i] <= a2[j]:
                    a.append(a1[i])
                    i += 1
                else:
                    a.append(a2[j])
                    j += 1
            elif i < n1:
                a.append(a1[i])
                i += 1
            else: #j < n2
                a.append(a2[j])
                j += 1
        return a

    def helper(array):
        n = len(array)
        if n == 1:
            res = array
        else:
            mid = n//2
            a1 = helper(array[0:mid])
            a2 = helper(array[mid:])
            res = merge(a1, a2)
        return res
    #print(merge([1,2,3], [0,1,3]))
    return helper(array)

def quick_sort(array):
    #Average Time complexity: O(nlogn) and Space complexity: O(n)
    #Not in-place version of quick-sort
    #Pivot can be: right end, left end, or randomly selected
    def merge(l,p,r):
        return l+p+r

    def helper(array):
        n = len(array)
        if n <= 1:
            return array
        else:
            l_array = []
            r_array = []
            p_array = []
            pivot = array[n-1] #using right end pivot
            for num in array:
                if num < pivot:
                    l_array.append(num)
                elif num == pivot:
                    p_array.append(num)
                else: #num > pivot
                    r_array.append(num)
            left = helper(l_array)
            right = helper(r_array)
            return merge(left, p_array, right)
    return helper(array)


def inplace_quick_sort(array):
    #Average Time complexity: O(nlogn) and Space complexity: O(1)
    #In-place version of quick-sort (space efficient)
    #Pivot can be: right end, left end, or randomly selected
    def helper(array, a, b):
        if a >= b: return None
        piv_id = b
        pivot = array[piv_id]
        left_id = a
        right_id = b-1
        while left_id <= right_id:
            while left_id <= right_id and array[left_id] < pivot:
                left_id += 1
            while left_id <= right_id and array[right_id] > pivot:
                right_id -= 1
            if left_id <= right_id:
                array[left_id], array[right_id] = array[right_id], array[left_id]
                left_id += 1
                right_id -= 1

        array[left_id], array[piv_id] = array[piv_id], array[left_id]
        helper(array, a, left_id-1)
        helper(array, left_id+1, b)
        return None
    helper(array, 0, len(array)-1)
    return array

import heapq
def heap_sort(array):
    #heapsort is very simple. It uses a heap data structure that can return n sorted elements in O(logn) time each
    n = len(array)
    heapq.heapify(array) #runs in O(nlog(n)) (heapifies inplace)
    res = []
    #This runs in O(nlogn) too
    for i in range(n):
        res.append(heapq.heappop(array))
    return res





array = [3,2,1,5,1,4,0, 2]
print(insertion_sort(array))
print(selection_sort(array))
print(bubble_sort(array))
print(merge_sort(array))
print(quick_sort(array))
print(inplace_quick_sort(array))
print(heap_sort(array))

print('\nArray Based Heap')

class ArrayBasedHeap():
    #min heap: where the root has the minimum value
    def __init__(self):
        self._data = []
        #idx(root) = 0
        #idx(left_child) = 2*idx(parent)+1
        #idx(right_child) = 2*idx(parent)+2

    def add(self, value):
        #Add value such that the heap is balanced and ordered
        #Two step process:
            #1. add value at the end of the array (to keep the heap balanced)
                #It will insert as the right most child at the lowest and if the lowest level has no empty nodes, then it will add as the left most child at the next level
                #the math just works out when doing data.append()!
            #2. Do UpHeap() to maintain the heap order
        self._data.append(value) #to keep the heap balanced
        self._UpHeap() #to keep maintain the heap order
        print(self._data)

    def _UpHeap(self):
        array = self._data
        n = len(array)-1
        while n > 0:
            p = (n-1)//2
            if array[n] < array[p]:
                array[p], array[n] = array[n], array[p]
                n = p
            else:
                break

    def delete(self):
        #Remove the minimum value from the heap (i.e. the root node) but make sure the heap is still balanced and ordered
        #Two step process:
            #1. Remove the last item from the array (bottom most leaf of the heap)
            #2. Put this value at the root
            #3. The do DownHeap() to maintain the heap order
        parent = 0
        n = len(self._data)
        min_val = self._data[parent]
        temp = self._data.pop()
        self._data[parent] = temp
        self._DownHeap()
        return min_val

    def _DownHeap(self):
        array = self._data
        parent = 0
        n = len(array)
        while parent < n:
            left_child = 2*parent + 1
            right_child = 2*parent + 2
            lc_val, rc_val = 9999, 9999 #some large number
            if left_child < n:
                lc_val = array[left_child]
            if right_child < n:
                rc_val = array[right_child]
            if lc_val < rc_val  and lc_val < array[parent]:
                array[parent], array[left_child] = array[left_child], array[parent]
                parent = left_child
            elif rc_val < lc_val and rc_val < array[parent]:
                array[parent], array[right_child] = array[right_child], array[parent]
                parent = right_child
            else:
                break
            print(array)
            

myHeap = ArrayBasedHeap()
myHeap.add(1)
myHeap.add(2)
myHeap.add(3)
myHeap.add(0)
myHeap.delete()





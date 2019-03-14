# test


class Node:
    def __init__(self):
        self.data = None
        self.next = None
     
    def setData(self,data):
        self.data = data
      
    def getData(self):
        return self.data
     
    def setNext(self,next):
        self.next = next
     
    def getNext(self):
        return self.next
        
class SinglyLinkedList:
    #constructor
    def __init__(self):
        self.head = None
        
    #method for setting the head of the Linked List
    def setHead(self,head):
        self.head = head
                     
    def arrange_in_pairs(self):
        temp = []
        node = self.head
        while node != None:
            temp.append(node.data)
            node = node.next

        n = len(temp)
        i,j = 0,n-1
        node_l = Node()
        node_l.data = temp[i]
        node_r = Node()
        node_l.next = node_r
        node_r.data = temp[j]
        self.head = node_l
        i += 1
        j -= 1
        while i < j:
            node_l = Node()
            node_r.next = node_l
            node_l.data = temp[i]
            node_r = Node()
            node_l.next = node_r
            node_r.data = temp[j]
            i+=1
            j-=1
        
        if i == j:
            node_l = Node()
            node_r.next = node_l
            node_l.data = temp[i]

        node = self.head
        while node != None:
            print(node.data)
            node=node.next


n1 = Node()
n2 = Node()
n3 = Node()
n4 = Node()
n5 = Node()
n6 = Node()
n1.data=1
n1.next = n2
n2.data=2
n2.next = n3
n3.data=3
n3.next = n4
n4.data=4
n4.next = n5
n5.data=5
n5.next = n6
n6.data=6

sll = SinglyLinkedList()
sll.head=n1
sll.arrange_in_pairs()

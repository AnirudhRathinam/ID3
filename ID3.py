import math
from random import randrange


class Queue:
    def __init__(self):
        self.item_list = []
        self.size = -1
    def isEmpty(self):
        return(self.size == -1)
    def push(self, s):
        self.item_list = [s] + self.item_list
        self.size+=1
    def pop(self):
        if(self.isEmpty()):
            return
        else:
            self.size-=1
            return
    def getTop(self):
        return self.item_list[self.size]
    def getSize(self):
        return self.size+1

class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.parent = None
        self.value = None
        self.att_id = None
        self.entropy = None
        self.data = data
        self.label = None
    def leftData(self):
        rows = len(self.data)
        left_data = []
        for i in range(0, rows):
            if(self.data[i][self.att_id] == '0'):
                left_data.append(data[i])
        return left_data
    def rightData(self):
        rows = len(self.data)
        right_data = []
        for i in range(0, rows):
            if(self.data[i][self.att_id] == '1'):
                right_data.append(data[i])
        return right_data


class Tree:
    def __init__(self):
        self.root = None
    def isEmpty(self):
        return (self.root == None)
    def add(self, node):
        if(self.root == None):
            self.root = node
    def getRoot(self):
        return self.root
    def delete(self):
        self.root = None
    def printBFS(self):
        pq = Queue()
        count = 0
        leafCount = 0
        pq.push(self.root)
        count+=1
        while(pq.isEmpty() == False):
            tmp = pq.getTop()
            if(pq.getTop().value != None):
                tval = pq.getTop().value
                #print(pq.getTop().value)
            else:
                if(pq.getTop().parent != None):
                    p, n = getPN(pq.getTop().data, pq.getTop().parent.att_id)
                if(p>n):
                    pq.getTop().label = 1
                    leafCount+=1
                else:
                    pq.getTop().label = 0
                    leafCount+=1
                #print('p, n: '+str(p)+' '+str(n))
                #print(pq.getTop().label)
            if(tmp.left != None):
                pq.push(tmp.left)
                count+=1
            if(tmp.right != None):
                pq.push(tmp.right)
                count+=1
            pq.pop()
        return(count, leafCount)
    def prune(self):
        current = self.root
        while(current.left.left != None):
            current = current.left
            class_data = [row[len(current.data[0])-1] for row in current.data]
            p = class_data.count('1')
            n = class_data.count('0')
        node = Node(current.data)
        if(p>n):
            node.label = 1
        else:
            node.label = 0
        node.parent = current.parent.left
        current.parent.left = node

def getColumn(data, att_id):
    return [row[att_id] for row in data]

def getPN(data, att_id):
    attribute_data = getColumn(data, att_id)
    return(attribute_data.count('1'), attribute_data.count('0'))

def _E(p, n):
    if(p == 0 or n == 0):
        return 0
    x = (p/(p+n))
    y = (n/(p+n))
    return(-1* (x*math.log2(x)) -1* (y*math.log2(y)))

def getPaths(node):
    path = []
    ref = []
    if(node.left == None and node.right == None):
        t_path = []
        current = node
        t_path.append(current.label)
        while(current.parent != None):
            if(current == current.parent.left):
                s = 0
            else:
                s = 1
            current = current.parent
            t_path.append(current.value+": " +str(s))
        path.append(t_path)
    if(node.left != None):
        path += (getPaths(node.left))
    if(node.right != None):
        path+=(getPaths(node.right))
    return path

def getDepth(node):
    path = getPaths(node)
    leafCount = len(path)
    total_depth = 0
    for i in range(0, leafCount):
        total_depth+=len(data[i])
    return(total_depth/leafCount), leafCount
    

def getRef(node):
    ref = []
    if(node.left == None and node.right == None):
        t_path = []
        current = node
        t_path.append(current.label)
        while(current.parent != None):
            if(current == current.parent.left):
                s = 0
            else:
                s = 1
            current = current.parent
            t_path.append(s)
        ref.append(t_path)
    if(node.left != None):
        ref += (getRef(node.left))
    if(node.right != None):
        ref+=(getRef(node.right))
    return ref

def calcPaths(node):
    path = []
    if(node.left == None and node.right == None):
        t_path = []
        current = node
        t_path.append(current.label)
        while(current.parent != None):
            if(current == current.parent.left):
                s = 0
            else:
                s = 1
            current = current.parent
            t_path.append(current.value)
        path.append(t_path)
    if(node.left != None):
        path += (calcPaths(node.left))
    if(node.right != None):
        path+=(calcPaths(node.right))
    return path

def getParams(data, att_id):
    attribute_data = getColumn(data, att_id)
    class_data = getColumn(data, len(data[0])-1)
    p = 0
    n = 0
    pp = 0
    pn = 0
    np = 0
    nn = 0
    for i in range (0, len(attribute_data)):
        if(attribute_data[i]=='1'):
            p+=1
            if(class_data[i]=='0'):
                pn+=1
            else:
                pp+=1
        else:
            n+=1
            if(class_data[i]=='0'):
                nn+=1
            else:
                np+=1
    return p, n, pp, pn, np, nn


def getEntropy(data, att_id):
    p, n, pp, pn, np, nn = getParams(data, att_id)
    return ((p/(p+n))*_E(pp,pn))+((n/(p+n)*_E(np,nn)))


def getPoints(data, pDat, nDat, pRef, nRef):
    points = 0
    for i in range(0, len(data)):
        if(class_list[i] == '0'):
            for j in range(0, len(nDat)):
                check = False
                for k in range(1, len(nDat[j])):
                    if(data[i][nDat[j][k]] != str(nRef[j][k])):
                        check = True
                if(check == False):
                    points+=1
        else:
            for j in range(0, len(pDat)):
                check = False
                for k in range(1, len(pDat[j])):
                    if(data[i][pDat[j][k]] != str(pRef[j][k])):
                        check = True
                if(check == False):
                    points += 1
        
    points = (points/len(data)) *100
    return points

def getRandomAttribute(tmp_ID_list):
    index = randrange(0, len(tmp_ID_list))
    return(tmp_ID_list[index])

def createRandom(data, tmp_ID_list, att_list, node, tree):
    if(node == None):
        root = Node(data)
        tree.add(root)
    else:
        root = node
    if(data == []):
        return
    else:
        p, n = getPN(data, len(data[0])-1)
    if(n == 0):
        #print('1')
        root.label = 1
        return root
    elif(p == 0):
        #print('2')
        root.label = 0
        return root
    elif(len(tmp_ID_list) == 0):
        if(p>n):
            #print('3')
            root.label = 1
            return root
        else:
            #print('4')
            root.label = 0
            return root
    else:
        q = Queue()
        q.push(root)
        while(len(tmp_list)!=0):
            root = q.getTop()
            node_id = getRandomAttribute(tmp_ID_list)
            val = att_list[node_id]
            tmp_list.remove(val)
            tmp_ID_list.remove(node_id)
            root.value = val
            root.att_id = node_id
            leftData = root.leftData()
            rightData = root.rightData()
            root.left = Node(leftData)
            root.left.parent = root
            root.right = Node(rightData)
            root.right.parent = root
            q.push(root.left)
            q.push(root.right)
            q.pop()



def getBestAttribute(tmp_ID_list, data):
    ent = 1
    node_id = 0
    for i in tmp_ID_list:
        _ent = getEntropy(data, i)
        #print(i, _ent)
        if(_ent < ent):
            ent = _ent
            node_id = i
    return node_id, ent

def ID3(data, tmp_ID_list, att_list, node, tree):
    if(node == None):
        root = Node(data)
        tree.add(root)
    else:
        root = node
    if(data == []):
        return
    else:
        p, n = getPN(data, len(data[0])-1)
    if(n == 0):
        #print('1')
        root.label = 1
        return root
    elif(p == 0):
        #print('2')
        root.label = 0
        return root
    elif(len(tmp_ID_list) == 0):
        if(p>n):
            #print('3')
            root.label = 1
            return root
        else:
            #print('4')
            root.label = 0
            return root
    else:
        q = Queue()
        q.push(root)
        while(len(tmp_list)!=0):
            root = q.getTop()
            node_id, ent = getBestAttribute(tmp_ID_list, data)
            val = att_list[node_id]
            tmp_list.remove(val)
            tmp_ID_list.remove(node_id)
            root.value = val
            root.att_id = node_id
            root.entropy = ent
            leftData = root.leftData()
            rightData = root.rightData()
            root.left = Node(leftData)
            root.left.parent = root
            root.right = Node(rightData)
            root.right.parent = root
            q.push(root.left)
            q.push(root.right)
            q.pop()

#Execution starts here
    
class_list = []
att_list = []
tmp_list = []
attId_list = []
tmp_ID_list = []
data = []
rows = 0
cols = 0

                                  
with open('train.dat', 'r') as file:
    attributes = file.readline()
    for i in attributes.split()[:-1]:
        att_list.append(i)
        tmp_list.append(i)
    cols = len(att_list)
    for l in file:
        rows+=1
        class_list.append(l.split()[cols])
        data.append(l.split())
for i in range (0, cols):
    attId_list.append(i)
    tmp_ID_list.append(i)

tree = Tree()
ID3(data, tmp_ID_list, att_list, None, tree)

print('\nID3 tree details')
print('-------------------\n')
count, leafCount = tree.printBFS()
print('\nThe paths from leaf to root are shown below: \n')
t = tree.getRoot()
p = getPaths(t)
n = len(p)
for i in range(0, n):
    print(p[i])

#Forming equations

p = calcPaths(t)
ref = getRef(t)

nodeData = getColumn(p, 0)
pDat = []
pRef = []
nDat = []
nRef = []
n = len(p)
for i in range(0, n):
    if(nodeData[i] == 0):
        nDat.append(p[i])
        nRef.append(ref[i])
    else:
        pDat.append(p[i])
        pRef.append(ref[i])

for i in range(0, len(pDat)):
    for j in range(1, len(pDat[i])):
        pDat[i][j] = att_list.index(pDat[i][j])
for i in range(0, len(nDat)):
    for j in range(1, len(nDat[i])):
        nDat[i][j] = att_list.index(nDat[i][j])

def check(data_row, nDat_row, nRef_row):
    flag = False
    for i in range(1, len(nDat_row)):
        if(str(nRef_row[i]) == data_row[nDat_row[i]]):
            flag = True
    if(flag == True):
        return True
    else:
        return False
points = getPoints(data, pDat, nDat, pRef, nRef)
depth, leafCount = getDepth(t)
print('\nID3 Accuracy:\n')
print('Number of training instances: ' + str(len(data)))
print('Number of training attributes: ' + str(len(att_list)-1))
print('Number of nodes in tree: ' + str(count))
print('Number of leaf nodes in the tree: ' + str(leafCount))
print('Accuracy in training data: ' + str(points))
print('\nID3 Tree:\tAvg depth: '+str(depth)+'\tNo. of leaves: '+str(leafCount))


print('\nRandom attribute selection tree details (tested for 5 trees)')
print('---------------------------------------------------------------\n')

for counter in range(0, 5):
    class_list = []
    att_list = []
    tmp_list = []
    attId_list = []
    tmp_ID_list = []
    data = []
    rows = 0
    cols = 0
                                  
    with open('train.dat', 'r') as file:
        attributes = file.readline()
        for i in attributes.split()[:-1]:
            att_list.append(i)
            tmp_list.append(i)
        cols = len(att_list)
        for l in file:
            rows+=1
            class_list.append(l.split()[cols])
            data.append(l.split())
    for i in range (0, cols):
        attId_list.append(i)
        tmp_ID_list.append(i)
    tree.delete()
    createRandom(data, tmp_ID_list, att_list, None, tree)
    count, leafCount = tree.printBFS()
    t = tree.getRoot()

    p = calcPaths(t)
    ref = getRef(t)

    nodeData = getColumn(p, 0)
    pDat = []
    pRef = []
    nDat = []
    nRef = []
    n = len(p)
    for i in range(0, n):
        if(nodeData[i] == 0):
            nDat.append(p[i])
            nRef.append(ref[i])
        else:
            pDat.append(p[i])
            pRef.append(ref[i])

    for i in range(0, len(pDat)):
        for j in range(1, len(pDat[i])):
            pDat[i][j] = att_list.index(pDat[i][j])
    for i in range(0, len(nDat)):
        for j in range(1, len(nDat[i])):
            nDat[i][j] = att_list.index(nDat[i][j])

    def check(data_row, nDat_row, nRef_row):
        flag = False
        for i in range(1, len(nDat_row)):
            if(str(nRef_row[i]) == data_row[nDat_row[i]]):
                flag = True
        if(flag == True):
            return True
        else:
            return False
    points = getPoints(data, pDat, nDat, pRef, nRef)
    depth, leafCount = getDepth(t)
    print('\nTree #: '+str(counter+1)+'\tDepth: '+str(depth)+'\tLeaves: '+str(leafCount)+'\tAccuracy: '+str(points))


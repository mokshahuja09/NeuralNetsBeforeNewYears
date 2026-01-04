r'''
Docstring for myTopoSort

So in this program we understand how the Toposort works. The topological sort
is an algorithm that works with acyclic graphs. As in, acyclic means that the
graph cannot return to where it began from. More specifically, the graph that 
this algorithm will typically work with is nodes that "pass on the potatoe and
never get it back". So for each node, the arrow must point forwards, and every 
point in the graph is acyclic.

Each node conists of two things: the node itself and its parents. So suppose we
have node C. Node C has value of 3, and the parents of node 3 are 1 and 4. Node
3 could be such that it is the parent of another node, say the Node D of value 5.
Maybe Node D could have 4 as a parent as well.

Now, note that 1, 4, are parents of 3, and 3 is a parent of 5. The list of nodes
that we have is S = [1, 3, 4, 5], and we want to sort it out such that for every
index i, j if i <= j then s_i <= s_j or s_i is not related to s_j. So s_i <= s_j
says that s_i is a parent of s_j. So if we organize the above list is organized
topologically, then the order is [1, 4, 3, 5], Since 1 is not a parent of 4, we 
could have equally has [4, 1, 3, 5], as it still satisfies the above relation. 
Another way to look at it is, for every i <= len(S), S[i] is a maximal element of the
subset list S[:i]. Let A be a set. x \in A is maximal if for every a \in A a !> x.

First Sort(Absolutely Useless for autograd):The way the code works is consider our list S. We iterate through the list, starting with
S[i], and then we iterate through all the elements after it, through S[j] and check whether it's a parent
of S[i], and if S[j] is a parent of S[i], then we swap the the two elements, that way 
the parent is before the child Node. We keep going until there's a list where we cannot find
a single node with its parents ahead of them. Each time the list is edited, we restart the double for loop.
I'm sure the code is inefficient, but it works.

Topo2Sort(BFS)(Functional for Autograd): Let me explain how this works by example: Note that this may be complicated
Suppose that 5 -> 2, 0; 2 -> 6, 3, 1; 6 -> 3; 3 -> 1 ;; 7 -> 4; 4 ->  0, 1. The topological order starting back from
1 is [7, 4, 5, 2, 6, 3, 1](Does not need to have 0 anywhere). Now clearly the final node that we end at is 1. So we start
with the dependancy list = [1]. Since 1 has parents = [2, 3, 4], we check if its already in dependancies, and since its not
we append all three [1, 2, 3, 4]. Now we go through the parents of 2, 3, 4: [5; 2, 6; 7]. Trivial with 5: [1, 2, 3, 4, 5].
Now, since 2 is already there in the list, we remove it, and re-add to the end to maintain the right ordering: [1, 3, 4, 5, 2, 6, 7]
Now we check the parents of [5; 2, 6; 7], which are: [None; 5, 2; None]. Now, for None, we just ignore and move forward. For 5, following
the same remove-append procedure from last time: [1, 3, 4, 2, 6, 7, 5]. Then for 2: [1, 3, 4, 6, 7, 5, 2]. Now we check parents of 2, 5: [5, None]
Then the final list is then: [1, 3, 4, 6, 7, 2, 5], which if you want in ascending is then: [5, 2, 7, 6, 4, 3, 1]. Note that there are other
ways of arranging this list, this is just based on the example above.

TopoSortDFS(Functional for autograd as well): So the point of this DFS is to search through a whole tree of parents before adding any to your
dependancy list. So following on from the same example above, let's start at 1. Note that I haven't yet added this to the dependancy list yet.
The way this works is that we dig down recursively to the bottom and stop when we find a leaf node, which we then add. Then the parents corresponding
to that branch get added subsequently, checking whether they've been added previously or not, to deal with duplicates. Since we do this branch wise and
depth first, we always add the lower ones in the branch before the upper ones, and thereby avoid creating any issues with the order.

Topo3Sort(DFS)(Best autograd implementation): This too, does a DFS to topologically sort the tensors. This works in a similar manner to the last one,
except much more efficiently. This algorithmn can be summarized in the following way: "add my parents first, and then add me". So this goes all the way
to the bottom, recursively by checking whether the your tensor/node has parents, and the then adds the first root node that does not have parents. Then
after all the root nodes of the corresponding branch have been added, the nodes above get added, and thereby in the same way as above, avoid the ordering
issue. One thing to note, is that this avoids branches that have been added altogether and is what makes this about 10 times faster the ToposortDFS algo
above. 

'''

import time

class Node:
    def __init__(self, num = 0, parents: list = None):
        self.num = num
        self.parents = parents
    
    
    def __repr__(self):
        return f"{self.num}"
    
def parentChecker(parentList: list[Node], myNode: Node, swapIdx):
    '''
    Docstring for parentChecker

    This program checks with a given node is a parent
    of another, and returns the index of the Node if it's
    a parent, and returns -1 if it isn't.
    
    :param parentList: List of parents for a Node
    :type parentList: list[Node]

    :param myNode: This is another node, not the child node
    :type myNode: Node

    :param swapIdx: The index of myNode, in a list outside.
    '''
    nodeIdx = -1
    for i in range(len(parentList)):
        if parentList[i].num == myNode.num:
            nodeIdx = swapIdx
        else:
            continue

    return nodeIdx

def TopoSort(myList: list[Node]):
    '''
    Docstring for TopoSort

    Topologically sorts acyclic graphs/lists
    
    :param myList: Description
    :type myList: list[Node]
    '''

    changed = True # This says, when we look through the list has the code changed the list?
    t = 0 # A counter to ensure the while loop doesn't break the code.

    while changed and (t < 100): # Runs as long as the list keeps getting edited
        
        changed = False
        t += 1


        for i in range(len(myList)): # Starts iteration of list
            edited = False # Similar to changed, except it starts the while loop faster
            currNode = myList[i] 
            parents = currNode.parents # Gets the parents of the parents

            if parents is None:
                print(f"{myList[i]} has no parents") # If it doesn't have parents we don't have to check for parents
                continue

            print(f"{myList[i]}'s parents are: {parents}\n\n")

            for j in range(i, len(myList)): # Starts look at the elements ahead of currNode
                
                swapNode = myList[j] # Gets the node that might be swapped

                swapIdx = parentChecker(parents, swapNode, j) # Checks if its a parent, if it is, we get a non-zero j > i.
                

                if swapIdx > i: # Checks if we got the index greater than current NOde's index
                    print(f"Index, Num to be swapped: {swapIdx}: {myList[swapIdx]} with {i}: {currNode}\n")
                    print("Swapped")
                    temp = myList[i] # Starts classic swap code
                    myList[i] = myList[swapIdx]
                    myList[swapIdx] = temp


                    print("Updated List:", myList, '\n\n')
                    edited = True # Since we edited the list, we changed the editied to true and break out of the first loop

                    break
            
            if edited:
                changed = True # Since edited, we now reset the changed to true to tell the while loop to keep going.
                break

def Topo2Sort(finalNode: Node):
    '''
    Docstring for Topo2Sort

    This topological sorting algorithm, although seems right, fails. Consider the case where you have the 'diamond' graph,
    where one node say z becomes z --> y <-- z. Now this should store two zs in the dependencies, however it will not, due
    to the part of the algorithm that does not allow duplicates. Unfortunate, and really large try, as this took me 2-3 hours
    to think of and build.

    Correction, this is me a few hours later, realizing that this is the expected behaviour that we want with out autograd! Yay!
    Although this is crazily inefficient, I could care less, since I came with this algo myself. I'll build the more efficient
    algorithm some other time.

    Okay. So why this works is because
    
    :param finalNode: Description
    :type finalNode: Node

    '''


    exists = True #Condition to exit the sort algorithm
    t = 0 # If everything else fails
    T = [finalNode] # Starts with the final Node of the graph, and initializes tensor list
    dependancies = [finalNode] # Initializes dependancy list

    while exists and (t < 100): 
        t += 1
        print("--------------------------\n")
        print(f"Now investigating the tensor list: {T}") 

        Tpar = [] # Initializes a parent list

        for each_T in T: # For loop to iterate through the tensors in the tensor list
            print("-----\n")
            print(f"Here's each Tensor: {each_T}\n\n")

            parents = each_T.parents # Gets parents of each tensor


            print(f"Each T's parents: {parents}")

            if parents is not None: # If it has parents, then we itereate through them and add them to the dependency list

                for eachParent in parents:

                    if eachParent not in dependancies: # If they're not in the dependency list, then add them at the end

                        dependancies.append(eachParent)
                        print(f"dependancies after {dependancies}\n")

                    else: # If the parent is in the dependancy list, then remove the parent, from wherever it is are and then add the parent to the end of the list
                        print(f"Found {eachParent} in {dependancies}")
                        print(f"dependancies before {dependancies}\n")

                        dependancies.remove(eachParent) # Removes the parent
                        print(f"dependancies inbetween {dependancies}\n")


                        dependancies.append(eachParent) # Adds the parent back to the list
                        print(f"dependancies after {dependancies}\n")
                    
                    if eachParent not in Tpar: # Adds the parents to the parent list
                        Tpar.append(eachParent)
                    else: # Does not allow duplicate parents, same way as above.
                        Tpar.remove(eachParent)
                        Tpar.append(eachParent)
                print("-----\n")
            else:
                continue

        print("--------------------------\n")
        if len(Tpar) > 0: # Checks if there exists a t in T such that T has a parent. If for every in t in T t does not have a parent, then break the loop.
            T = Tpar
            exists = True
        else:
            exists = False

    return dependancies


def TopoSortDFS(finalNode: Node):
    visited = set() # Set is needed because we need to do lookups constantly, and sets are the fastest way to do it
    topoSorted = [] # Initializing the final dependancy and sorted list

    def buildTopo(childNode: Node, visitedSet: set, topoSortList: list):

        parents = childNode.parents

        if (parents is None) and (childNode not in visited): # Checks if this is a leaf node, and whether or not it is already in the topoSort list
            visited.add(childNode) # If it doesn't have parents and it's not already in the toposort list, then the function adds it to both
            topoSortList.append(childNode)
        
        elif parents is not None: # However, if it has parents, we need to re-do it with the gradparents as well, until we find a leaf node.

            for parent in parents: # Go through each parent, and recursively keep going until you find a leaf node, and build up from there
                buildTopo(parent, visitedSet = visitedSet, topoSortList= topoSortList)

                if parent not in visitedSet: # Now that the children of each of the parents has been added, we can now check if the parents can be added as well
                    visitedSet.add(parent)
                    topoSortList.append(parent)

    buildTopo(childNode= finalNode, visitedSet= visited, topoSortList= topoSorted)
    topoSorted.append(finalNode) # Since there was no call to add this to the end of the list, we add it now

    return topoSorted 

def Topo3Sort(finalNode: Node):
    visited = set() # Used for fast lookups
    topoSorted = [] # Final dependancy list

    def buildTopo(childNode: Node, visitedSet: set, topoSortList: list):

        if childNode in visitedSet: # If our childNode is in visited, we don't need to check that branch, as its already covered
            return
        
        if childNode.parents is not None: # If there are parents keep going until you find one without parents
            for parent in childNode.parents:
                buildTopo(parent, visitedSet= visitedSet, topoSortList= topoSortList)
        
        # The first time this part of the function will run will be when the code reaches a leaf node with no parents,
        # And after that this will exeute when the branch keeps completing, like when the for loop for each node finishes.
        visitedSet.add(childNode) 
        topoSortList.append(childNode)


    buildTopo(childNode= finalNode, visitedSet= visited, topoSortList= topoSorted)

    return topoSorted 


H = Node(7)
F = Node(5)
E = Node(4, parents= [H])
C = Node(2, parents = [F])
G = Node(6, parents = [C])
D = Node(3, parents= [C, G])
B = Node(1, parents = [E, D])
A = Node(0, parents= [E, F])

startBFSTime = time.time()

BFS = Topo2Sort(B)

endBFSTime = time.time()

startDFS_otherTime = time.time()

DFS_other = TopoSortDFS(B)

endDFS_otherTime = time.time()


startDFS_mainTime = time.time()

DFS_main = Topo3Sort(B)

endDFS_mainTime = time.time()

print(BFS)
print(f"BFS's time = {startBFSTime - endBFSTime}")

print(DFS_other)
print(f"DFS's (mine) time = {startDFS_otherTime - endDFS_otherTime}")

print(DFS_main)
print(f"DFS's (main) time = {startDFS_mainTime - endDFS_mainTime}")




# nodeList = [A, B, C, D, E, F]

# print(nodeList)




# print(nodeList)



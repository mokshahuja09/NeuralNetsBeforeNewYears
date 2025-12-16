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

The way the code works is consider our list S. We iterate through the list, starting with
S[i], and then we iterate through all the elements after it, through S[j] and check whether it's a parent
of S[i], and if S[j] is a parent of S[i], then we swap the the two elements, that way 
the parent is before the child Node. We keep going until there's a list where we cannot find
a single node with its parents ahead of them. Each time the list is edited, we restart the double for loop.
I'm sure the code is inefficient, but it works.

'''

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
    Although this is crazilly inefficient, I could care less, since I came with this algo myself. I'll build the more efficient
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

        for each_T in T: # FOr loop to iterate through the tensors in the tensor list
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


H = Node(7)
F = Node(5)
E = Node(4, parents= [H])
C = Node(2, parents = [F])
G = Node(6, parents = [C])
D = Node(3, parents= [C, G])
B = Node(1, parents = [E, D])
A = Node(0, parents= [E, F])



X = Node(4)
# nodeList = [A, B, C, D, E, F]

# print(nodeList)


print(Topo2Sort(B))

# print(nodeList)



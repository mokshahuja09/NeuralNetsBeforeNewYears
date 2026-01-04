# Update: 2026|01|04: Optimized the Toposort Algorithm to be faster by replacing the original BFS(Breadth-First Search) with a DFS(Depth-First Search)
#                     The optimization was great! Sped up my neural net calculation from minutes to seconds! Perfect!

# Next Steps: Documentation for the whole thing, plus a write up of what i learnt maybe.

import numpy as np
from functools import partial
import matplotlib.pyplot as plt

def myPrint(statement = f"Demo Day", supress = True):
        if not supress:
            print(statement)

#----------------------------------------- Tensor Class --------------------------------------------
class Tensor:
    def __init__(self, x: np.ndarray, grad = None, parents = None , grad_fn = None):

        self.data = x # Data array

        self.grad = grad # partial of function f with respect to x: df/dx

        self.parents = parents # The parent tensors that created this output

        self.grad_fn = grad_fn # Function through which parent tensors were fed to create this tensor

    def backwards(self):
        '''
            initiates the backward pass to compute the gradient of a function
            with respect to the initial variables.
        '''

        def Topo2Sort(finalNode: Tensor):

            exists = True #Condition to exit the sort algorithm
            t = 0 # If everything else fails
            T = [finalNode] # Starts with the final Node of the graph, and initializes tensor list
            dependancies = [finalNode] # Initializes dependancy list

            while exists and (t < 1000): 
                t += 1
                myPrint(statement = "--------------------------\n")
                myPrint(f"Now investigating the tensor list: {T}") 

                Tpar = [] # Initializes a parent list

                for each_T in T: # For loop to iterate through the tensors in the tensor list
                    myPrint("-----\n")
                    myPrint(f"Here's each Tensor: {each_T}\n\n")

                    parents = each_T.parents # Gets parents of each tensor


                    myPrint(f"Each T's parents: {parents}")

                    if parents is not None: # If it has parents, then we itereate through them and add them to the dependency list

                        for eachParent in parents:

                            if eachParent not in dependancies: # If they're not in the dependency list, then add them at the end

                                dependancies.append(eachParent)
                                myPrint(f"dependancies after {dependancies}\n")

                            else: # If the parent is in the dependancy list, then remove the parent, from wherever it is are and then add the parent to the end of the list
                                myPrint(f"Found {eachParent} in {dependancies}")
                                myPrint(f"dependancies before {dependancies}\n")

                                dependancies.remove(eachParent) # Removes the parent
                                myPrint(f"dependancies inbetween {dependancies}\n")


                                dependancies.append(eachParent) # Adds the parent back to the list
                                myPrint(f"dependancies after {dependancies}\n")
                            
                            if eachParent not in Tpar: # Adds the parents to the parent list
                                Tpar.append(eachParent)
                            else: # Does not allow duplicate parents, same way as above.
                                Tpar.remove(eachParent)
                                Tpar.append(eachParent)
                        myPrint("-----\n")
                    else:
                        continue

                myPrint("--------------------------\n")
                if len(Tpar) > 0: # Checks if there exists a t in T such that T has a parent. If for every in t in T t does not have a parent, then break the loop.
                    T = Tpar
                    exists = True
                else:
                    exists = False

            return dependancies
        
        def Topo3Sort(finalNode: Tensor):
            '''

                DFS Topological Sort

                Topo3Sort(DFS)(Best autograd implementation): This too, does a DFS to topologically sort the tensors. This algorithmn can be summarized in the following way:
                "add my parents first, and then add me". So this goes all the way to the bottom, recursively by checking whether the your tensor/node has parents, and the 
                then adds the first root node that does not have parents. Then after all the root nodes of the corresponding branch have been added, the nodes above get added,
                and thereby in the same way as above, avoid the ordering issue. One thing to note, is that this avoids branches that have been added altogether,
            
            '''
            visited = set()
            topoSorted = []

            def buildTopo(childNode: Tensor, visitedSet: set, topoSortList: list):

                if childNode in visitedSet:
                    return
                
                if childNode.parents is not None:
                    for parent in childNode.parents:
                        buildTopo(parent, visitedSet= visitedSet, topoSortList= topoSortList)
                
                visitedSet.add(childNode)
                topoSortList.append(childNode)


            buildTopo(childNode= finalNode, visitedSet= visited, topoSortList= topoSorted)

            return topoSorted
        
        myPrint("\nGetting Dependencies\n", supress= True)

        dependencies = Topo3Sort(self)

        myPrint(f"Here is the dependency list of tensors :{dependencies}", supress= True)

        # Starts the backward pass through the dependency list.
        for i in range(len(dependencies)):

            ith_Tensor = dependencies[len(dependencies) - i - 1]

            #If the tensor has parents and the has a grad_fn, then call the backwards function corresponding to it
            if (ith_Tensor.grad_fn is not None) and (ith_Tensor.parents is not None):
                if ith_Tensor.grad is None:
                    ith_Tensor.grad = np.ones_like(ith_Tensor.data)
                    ith_Tensor.grad_fn(ith_Tensor.parents, ith_Tensor.grad) # Calls the backwards pass function to calculate the gradients for the input Tensor.

                    myPrint(f"Had no grad, so set the ith Tensor's: {ith_Tensor} grad to 1.")
                else:
                    ith_Tensor.grad_fn(ith_Tensor.parents, ith_Tensor.grad)
                    myPrint(f"The grad for the ith tensor: {ith_Tensor} exists, so continued to calculate the gradient")
            else:
                myPrint(f"This tensor: {ith_Tensor} does not have parents or does not have a grad function")

        myPrint(f"Completed backwards pass for {self}")

    def __add__(self, y):

        if isinstance(y, (int, float)):
            y = Tensor(x=np.array(y))


        output = self.data + y.data

        outputTensor = Tensor(x= output, parents = [self, y], grad_fn= addBackwards)

        return outputTensor

    def __sub__(self, y):

        if isinstance(y, (int, float)):
            y = Tensor(x=np.array(y))

        output = self.data - y.data

        outputTensor = Tensor(x= output, parents = [self, y], grad_fn= subBackwards)

        return outputTensor

    def __mul__(self, y):
        ''' 
            Function that allows for multiplication between tensors. Creates a tensor output.
        '''
        if isinstance(y, (int, float)):
            y = Tensor(x=np.array(y))
        
        output = self.data * y.data

        outputTensor = Tensor(x = output, parents = [self, y], grad_fn= mulBackwards)

        return outputTensor
    
    def __rmul__(self, y):
        return self * y
    
    def __pow__(self, n):

        output = self.data ** n

        grad_fn_with_power = partial(powBackwards, power = n)
        outputTensor = Tensor(x = output, parents = [self], grad_fn=grad_fn_with_power)

        return outputTensor

    def dot(self, X):
        '''
        Docstring for dot

        Basically takes a do product of your tensor and another tensor
        
        :param self: Description
        :param X: Description


        '''

        output = self.data.dot(X.data)

        outputTensor = Tensor(x = output, parents = [self, X], grad_fn = dotBackwards)

        return outputTensor

    def __matmul__(self, Y):
        X = self.data
        Y_data = Y.data

        output = X @ Y_data
        outputTensor = Tensor(output, parents = [self, Y], grad_fn = matmulBackwards)

        return outputTensor
        
    def ReLu(self):
        output = np.maximum(0, self.data)

        return Tensor(output, parents = [self], grad_fn= ReLuBackwards)    


    def softmax(self):
        logits_shifted = self.data - np.max(self.data, axis=0, keepdims=True)
        exps = np.exp(logits_shifted)
        probs = exps / np.sum(exps, axis=0, keepdims=True)

        return Tensor(probs)

    def softmaxCrossEntropy(self, Y):
        # Here, our self.data might be an m x n matrix with all our weighted ouputs from the neural network
        # So for each value in a column vector, we will exponentiate the ith value, and then divide by the sum of all exponentiated values of the column.
        # so p_i1 = e^(z_i1)/sum_over_j(e^(z_j1))

        logits_shifted = self.data - np.max(self.data, axis=0, keepdims=True)
        exps = np.exp(logits_shifted)
        probs = exps / np.sum(exps, axis=0, keepdims=True)

        # Now, our losses can be found, by using all our one hot y vectors by doing an element-wise multiplication
        losses = Y.data * np.log(probs + 1e-15)

        # Then here, we will sum the losses down each column, collapsing it into a 1 x n matrix
        individual_losses = -np.sum(losses, axis=0)

        # Then here, we average the losses to ensure that when its added up later on we don't get something too too big.
        output_averaged = np.mean(individual_losses)
        
        # We store this so that we can use this in the backwards pass for softmax-CrossEntropy.
        self.probs = probs
        output_Tensor = Tensor(output_averaged, parents = [self, Y], grad_fn= softmaxCrossEntropyBackwards)


        return output_Tensor

    def mean(self):
        # We need to divide the data by the number of elements
        div = self.data.size
        output = np.mean(self.data)
        
        # Backward logic: 
        # The gradient of mean(x) w.r.t x is 1/N for every element
        def meanBackwards(parents, childGrad):
            x = parents[0]
            # childGrad is a scalar (1.0)
            # We broadcast it to 1/N for every element in x
            d_x = (1.0 / x.data.size) * childGrad * np.ones_like(x.data)
            accGrad(x, d_x)

        return Tensor(output, parents=[self], grad_fn=meanBackwards)

    def __repr__(self):
        return f"{self.data}"
 

#----------------------------------------- Backwards Functions --------------------------------------------

def unbroadcast(childGrad: np.ndarray, reqGradShape):

    '''
    Docstring for unbroadcast

    This function will be used to when we return gradients of sizes large than what is 
    required. As in, in linear regression, the gradient that will be returned through
    backwards pass is actually a vector, and we need the gradient with respect to b1 and
    b0 to be scalar derivatives basically.
    
    :param childGrad: Description
    :param parentShape: Description

    '''

    
    while childGrad.ndim > len(reqGradShape):
        childGrad = np.sum(childGrad, axis = 0)

    for i, dim in enumerate(reqGradShape):
        if dim == 1:
            childGrad = np.sum(childGrad, axis = i, keepdims= True)
    
    return childGrad

def accGrad(x: Tensor, curr_grad):
    ''' 
        Updates the gradient of a tensor.
    '''
    if x.grad is None:
        x.grad = curr_grad
    else:
        x.grad += curr_grad

def powBackwards(parents: list[Tensor],  childGrad, power = 2):
    x = parents[0]

    d_x = (power * x.data**(power - 1)) * childGrad

    if d_x.shape != x.data.shape:
        d_x = unbroadcast(d_x, x.data.shape)

    accGrad(x, d_x)

def addBackwards(parents: list[Tensor, Tensor], childGrad):
    x = parents[0]
    y = parents[1]


    d_x = 1 * childGrad
    d_y = 1 * childGrad

    if d_x.shape != x.data.shape:
        d_x = unbroadcast(d_x, x.data.shape)
        
    if d_y.shape != y.data.shape:
        d_y = unbroadcast(d_y, y.data.shape)

    accGrad(x, d_x)
    accGrad(y, d_y)

def subBackwards(parents: list[Tensor, Tensor], childGrad):
    x = parents[0]
    y = parents[1]


    d_x = 1 * childGrad
    d_y = - 1 * childGrad

    if d_x.shape != x.data.shape:
        d_x = unbroadcast(d_x, x.data.shape)
        
    if d_y.shape != y.data.shape:
        d_y = unbroadcast(d_y, y.data.shape)

    accGrad(x, d_x)
    accGrad(y, d_y)

def mulBackwards(parents: list[Tensor, Tensor], childGrad):
    x = parents[0]
    y = parents[1]


    '''
        Here we basically take the partials with respect to each parent tensor: 
        x, y. We have f(x, y) = xy => df/dx = y and df/dy = x. Note that the 
        gradient of a child or downstream function of f, say like g(f) might 
        have another derivative, and so we have multiply: dg/df * df/dx and 
        dg/df * df/dy. That's what childGrad is.
    '''

    myPrint(f"y data: {y.data}")
    myPrint(f"child grad =  {childGrad}")


    d_x = y.data * childGrad
    d_y = x.data * childGrad

    if d_x.shape != x.data.shape:
        d_x = unbroadcast(d_x, x.data.shape)
        
    if d_y.shape != y.data.shape:
        d_y = unbroadcast(d_y, y.data.shape)

    myPrint(f"Multiplication Grads:")
    myPrint(f"df/dx = {d_x}")
    myPrint(f"df/dy = {d_y}\n\n")


    accGrad(x, d_x)
    accGrad(y, d_y)

def dotBackwards(parents: list[Tensor, Tensor], childGrad):

    X = parents[0]
    Y = parents[1]

    d_X = Y.data * childGrad
    d_Y = X.data * childGrad

    accGrad(X, d_X)
    accGrad(Y, d_Y)

def matmulBackwards(parents: list[Tensor, Tensor], childGrad):
    W = parents[0] # This is an k x n matrix
    X = parents[1] # This is n x p
    # Then clearly, through matmul, we get back a k x p matrix
    # Since we get a k x p matrix, our incoming grad must also be k x p
    # So childGrad: k x p

    # Order matters, if Z = XY then dZ/dX is X^(T)
    # Note that we're trying to update the weights in X,
    # So we need d_X to be of the same dimension: k x n
    # Since childGrad is k x p and Y.T is p x n, we clearly get back k x n
    d_W = childGrad @ X.data.T


    # Similarly, here we need d_Y to be of dimension: n x p
    # X.T is clearly n x k and child grad is k x p
    d_X = W.data.T @ childGrad


    if d_W.shape != W.data.shape:
        d_W = unbroadcast(d_W, W.data.shape)
    if d_X.shape != X.data.shape:
        d_X = unbroadcast(d_X, X.data.shape)


    # Now we need to do accGrad
    accGrad(W, d_W)
    accGrad(X, d_X)

def ReLuBackwards(parents: list[Tensor, Tensor], childGrad):
    X = parents[0]

    myMask = (X.data > 0).astype(float)
    
    d_X = childGrad * myMask
    
    accGrad(X, d_X)

def softmaxCrossEntropyBackwards(parents: list[Tensor], childGrad):

    Z = parents[0]
    Y = parents[1]

    P = Z.probs
    batchSize = Y.data.shape[1]
    
    d_Z = (P - Y.data)/batchSize * childGrad

    accGrad(Z, d_Z)

def zeroGrad(tensors: list[Tensor]):
    for i in range(len(tensors)):
        tensors[i].grad = None

def gradUpdate(tensors: list[Tensor], lr):
    grad_mag = 0
    for i in range(len(tensors)):
        grad_mag += np.sum(tensors[i].grad ** 2)
        myPrint(f"Tensor: {i} Grad: \n{tensors[i].grad}", supress = True)
        tensors[i].data -= tensors[i].grad * lr
    return grad_mag
        # print(tensors[i], '\n')


if __name__ == "__main__":

    # Executing the XOR neural net, with 2 neurons in the first layer and 1 in the output layer.
    print('hello world')

    tensorList = []

    X_data = np.array( [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]  ) # Create data
    Y_data = np.array([[1.0, 1.0, 0, 0]]) # Label outputs
    W_1_data = np.array( [[1.0, -1.0], [1.0, 1.0]]) # Initialize the weight matrices. Note that choosing negative numbers ensures that not all neurons are firing.
    W_2_data = np.array( [[1.0, 1.0]])

    b_1_data = np.array([[-0.5], [-0.5]]) # For the same reason not choosing all postive numbers for the biases ensures that neurons aren't always firing
    b_2_data = np.array(0.0)

    # Tensorize all the data, weights and baises
    X = Tensor(X_data)
    W_1 = Tensor(W_1_data)
    W_2 = Tensor(W_2_data)
    Y = Tensor(Y_data)

    b1 = Tensor(b_1_data)
    b2 = Tensor(b_2_data)

    # Create the list of tensors that needs to be updated with their respective grads.
    tensorList = [ W_1, W_2, b1, b2]

    lr = 0.05

    for i in range(2000):

        # Layer 1
        Z_1 = (W_1 @ X) + b1
        A_1 = Z_1.ReLu()
        
        # Layer 2 (Output layer)
        Z_2 = W_2 @ A_1 + b2

        # MSE used for this data set
        SquaredErrors = (Z_2 - Y)**2
        L = SquaredErrors.mean()

        print(f"MSE Loss: {L.data}")

        L.backwards()

        # Update all the tensors and calculate whether the grad is small enough to quit.
        grad_mag= gradUpdate(tensorList, lr = lr)
        if grad_mag < 0.00005:
            print(f"broke out of the loop due to tiny grads: {round(grad_mag, 6)}")
            break
        zeroGrad(tensorList)

    print(f"Here is W_1:\n{W_1}\n\n")
    print(f"Here is b_1:\n{b1}\n\n")
    print(f"Her is W_2: \n{W_2}\n\n") 
    print(f"Here is b_2:\n{b2}\n\n")

    newData = np.array([[1], [0]])

    inputTensor = Tensor(newData)

    Z1 = ((W_1 @ inputTensor) + b1).ReLu()

    print(inputTensor.data.shape)
    
    output = W_2@Z1 + b2
    print(output)

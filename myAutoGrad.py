# Note: This is not the whole autograd, still building the whole thing, there are problems: specifically the topological sort has not been implemented yet.
# An example of where this code fails is: y = z^(2), such that z = 2x

# Update: Tue, 2025|12|16, the code finally has a base to rest on: the topological sort. Although inefficienct, it sorts through the graph perfectly, and
#         now the all that is required is coming up with the right forwards and backwards passes for the elementary functions, and we should be good to go.

import numpy as np
from functools import partial

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

            while exists and (t < 100): 
                t += 1
                myPrint(statement = "--------------------------\n")
                myPrint(f"Now investigating the tensor list: {T}") 

                Tpar = [] # Initializes a parent list

                for each_T in T: # FOr loop to iterate through the tensors in the tensor list
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
        
        myPrint("Getting Dependencies", supress= False)

        dependencies = Topo2Sort(self)

        myPrint(f"Here is the dependency list of tensors :{dependencies}", supress= False)

        # Starts the backward pass through the dependency list.
        for i in range(len(dependencies)):

            ith_Tensor = dependencies[i]

            #If the tensor has parents and the has a grad_fn, then call the backwards function corresponding to it
            if (ith_Tensor.grad_fn is not None) and (ith_Tensor.parents is not None):
                if ith_Tensor.grad is None:
                    ith_Tensor.grad = 1
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

if __name__ == "__main__":
    x_array = np.array([1, 2, 3])
    y_array = np.array([1, 2, 2])

    b0 = Tensor(x = np.array(1.0))
    b1 = Tensor(x = np.array(1.0))

    X = Tensor(x = x_array)
    Y = Tensor(x = y_array)

    lr = 0.01

    for i in range(1000):
        I = (Y - (b0 + (b1 * X)))
        avg_Const = 1/(len(X.data))
        SE = I.dot(I)

        MSE = avg_Const * SE

        print(SE)

        SE.backwards()

        print(f"b0's grad = {b0.grad}")
        print(f"b1's grad = {b1.grad}")

        if (np.sqrt(b0.grad**2 + b1.grad**2) < 0.0005):
            print('Breaking due to tiny gradients')
            break

        b0.data -= b0.grad * lr
        b1.data -= b1.grad * lr

        b0.grad = 0
        b1.grad = 0

    print(b0, b1)
    

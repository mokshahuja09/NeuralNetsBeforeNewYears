import numpy as np
import myAutoGrad as ag



class Module:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        return []
    
    def zeroGrad(self):
        for tensor in self.parameters():
            tensor.grad = None

        
class SGD:
    def __init__(self, params: list[ag.Tensor], lr):
        self.params = params
        self.lr = lr
    
    def step(self):
        for eachTensor in self.params:
            if eachTensor.grad is not None:
                eachTensor.data -= self.lr * eachTensor.grad
    
    def zeroGrad(self):
        for eachTensor in self.params:
            eachTensor.grad = None


    
class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        scale = np.sqrt(2.0 / in_features)

        self.W = ag.Tensor(np.random.randn(out_features, in_features) * scale) #np.random.randn
        self.b = ag.Tensor(np.zeros((out_features, 1)))
    
    def forward(self, X: ag.Tensor):
        if X.data.shape[0] != self.in_features:
            raise AttributeError(f"Data shape: {X.data.shape[0]} does not match required feature shape: {self.in_features}")
        
        Z = (self.W @ X) + self.b

        return Z
    
    def parameters(self):
        return [self.W, self.b]



    
class Sequential(Module):

    def __init__(self, layers: list):
        super().__init__()
        self.layers = layers
    
    def forward(self, X):
        layers = self.layers
        input_data = X

        for eachLayer in layers:
            input_data = eachLayer(input_data)
        
        outputTensor= input_data

        return outputTensor
        
    def parameters(self):
        layers = self.layers
        tensorList = []

        for eachLayer in reversed(layers):
            tensorList += eachLayer.parameters()
        
        return tensorList


class ReLu(Module):
    
    def forward(self, Z: ag.Tensor):
        return Z.ReLu()


            
if __name__ == '__main__':
    X_data = np.array( [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]  )
    Y_data = np.array([[1.0, 1.0, 0, 0]])

    X = ag.Tensor(X_data)
    Y = ag.Tensor(Y_data)

    layers = Sequential([Linear(2, 8), ReLu(), Linear(8, 1)])

    optimizer = SGD(params = layers.parameters(), lr = 0.05)

    for i in range(2000):
        output = layers(X)

        diff = (output - Y)
        loss = (diff * diff).mean()

        print(f"Loss for the {i}th iteration: {loss.data: 2f}")

        loss.backwards()

        optimizer.step()
        optimizer.zeroGrad()

    # np.set_printoptions(precision=2, suppress=True)
    print(f"final: {layers(X).data}")


        


    


    

    



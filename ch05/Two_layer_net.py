import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params={}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
        for layer in self.layers.values():
            print(layer)

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
    
class FourLayerNet(TwoLayerNet):
    def __init__(self, input_size, hidden_size, hidden_size2, hidden_size3, output_size, weight_init_std=0.01):
        self.params={}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, hidden_size2)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size2, hidden_size3)
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size3, output_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['b3'] = np.zeros(hidden_size3)
        self.params['b4'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu() 
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.lastLayer = SoftmaxWithLoss()

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db
        grads['W4'] = self.layers['Affine4'].dW
        grads['b4'] = self.layers['Affine4'].db

        return grads
    
class ThreeLayerNet(TwoLayerNet):
    def __init__(self, input_size, hidden_size, hidden_size2, hidden_size3, hidden_size4, hidden_size5, output_size, weight_init_std=0.01):
        self.params={}
        concat_layers = [input_size].append(hidden_size).append(output_size)
        for i in range(len(concat_layers)):
            self.params['W'+str(i+1)] = weight_init_std * np.random.randn(concat_layers[i], concat_layers[i+1])
            self.params['b'+str(i+1)] = weight_init_std * np.random.randn(concat_layers[i+1])

        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, hidden_size2)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size2, hidden_size3)
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size3, hidden_size4)
        self.params['W5'] = weight_init_std * np.random.randn(hidden_size4, hidden_size5)
        self.params['W6'] = weight_init_std * np.random.randn(hidden_size5, output_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['b3'] = np.zeros(hidden_size3)
        self.params['b4'] = np.zeros(hidden_size4)
        self.params['b5'] = np.zeros(hidden_size5)
        self.params['b6'] = np.zeros(output_size)

        self.layers = OrderedDict()
        for i in range(1, len(concat_layers) - 1):
            self.layers['Affine'+str(i)] = Affine(self.params['W'+str(i)], self.params['b'+str(i)])
            self.layers['Relu'+str(i)] = Relu()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Relu4'] = Relu()
        self.layers['Affine5'] = Affine(self.params['W5'], self.params['b5'])
        self.layers['Relu5'] = Relu()
        self.layers['Affine6'] = Affine(self.params['W6'], self.params['b6'])
        self.lastLayer = SoftmaxWithLoss()

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db
        grads['W4'] = self.layers['Affine4'].dW
        grads['b4'] = self.layers['Affine4'].db
        grads['W5'] = self.layers['Affine5'].dW
        grads['b5'] = self.layers['Affine5'].db
        grads['W6'] = self.layers['Affine6'].dW
        grads['b6'] = self.layers['Affine6'].db
        return grads
    

class AnyLayerNet(TwoLayerNet):
    def __init__(self, input_size, output_size, *hidden_size, weight_init_std=0.1):
        print(output_size, input_size, hidden_size)
        self.params={}
        concat_layers = list(hidden_size)
        concat_layers.insert(0, input_size)
        concat_layers.append(output_size)
        print(concat_layers)
        self.para_num = len(concat_layers)
        for i in range(1, self.para_num):
            self.params['W'+str(i)] = weight_init_std * np.random.randn(concat_layers[i-1], concat_layers[i])
            self.params['b'+str(i)] = weight_init_std * np.random.randn(concat_layers[i])

        self.layers = OrderedDict()
        for i in range(1, self.para_num):
            self.layers['Affine'+str(i)] = Affine(self.params['W'+str(i)], self.params['b'+str(i)])
            if i != self.para_num - 1:
                self.layers['Relu'+str(i)] = Relu()
        for layer in self.layers.values():
            print(layer)
        self.lastLayer = SoftmaxWithLoss()

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for i in range(1, self.para_num):
            grads['W'+str(i)] = self.layers['Affine'+str(i)].dW
            grads['b'+str(i)] = self.layers['Affine'+str(i)].db

        return grads
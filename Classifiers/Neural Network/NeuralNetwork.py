import numpy as np;
import math;
from numpy.linalg import inv;
from random import shuffle;
import Reader;

def Predict(item,Weights,hidden,output):
    W1, W2 = Weights;
    
    item = np.append(1,item); #Augment feature vector

    #_Forward Propagation_#
    outputHidden = LayerActivation(hidden,W1,item);
    outputHidden = np.append(1,outputHidden);
    outputFinal = LayerActivation(output,W2,outputHidden);

    m = outputFinal[0];
    index = 0;
    for i in range(1,len(outputFinal)):
        output = outputFinal[i];
        
        if(output > m):
            m = output;
            index = i;

    #Initialize prediction vector to zeros
    y = [0 for i in range(len(outputFinal))];
    y[index] = 1; #Set guessed class to 1

    return y; #Return prediction vector

def Accuracy(X,Y,Weights,hidden,output):
    correct = 0;
    
    for i in range(len(X)):
        x = X[i];
        y = list(Y[i].A1);

        guess = Predict(x,Weights,hidden,output);

        print y,guess;

        if(y == guess):
            correct += 1;

    print correct/float(len(X));

###_Auxiliary Functions_###
def Sigmoid(x):
    return 1 / (1 + np.exp(-x));

def SigmoidDerivative(x):
    return np.multiply(x,1-x);

def InitializeWeights(f,hidden,output):
    #Initialize weights with random values in [0,1] (including bias)
    #s : length of second layer

    #Augment feature vectors with bias
    f += 1;

    #Initialize second (hidden) layer weights
    W1 = [[np.random.uniform() for i in range(f)] for j in range(hidden)];
    W1 = np.matrix(W1);

    #Initialize third (output) layer weights
    W2 = [[np.random.uniform() for i in range(hidden+1)] for j in range(output)];
    W2 = np.matrix(W2);

    return W1, W2;

###_Core Functions_###
def NeuronActivation(Weight,Input):
    #Sum = 0;
    #for i in range(len(Input)):
    #    Sum += Input[i]*Weight[i];

    Sum = Weight*Input.T;
    
    #Return sigmoid, turn numpy array to number
    activation = Sigmoid(Sum);
    activation = np.asscalar(activation);
    
    return activation;

def LayerActivation(Size,Weights,Input):
    layerOutput = [0]*Size;
    
    for i in range(Size):
        layerOutput[i] = NeuronActivation(Weights[i].A1,np.matrix(Input));

    return layerOutput;

def Train(X,Y,f,hidden,output,W1,W2,r):
    for i in range(len(X)):
        x = X[i];
        y = Y[i].A1;
        x = np.append(1,x); #Augment feature vector

        #_Forward Propagation_#        

        activationHidden = LayerActivation(hidden,W1,x);
        #Augment hidden activation output
        augmentedHidden = np.append(1,activationHidden);
        outputFinal = LayerActivation(output,W2,augmentedHidden);

        #_Backpropagation_#
        errorOutput = np.matrix(y - outputFinal);
        
        errorHidden = np.dot(W2.T,np.matrix(errorOutput).T)*SigmoidDerivative(augmentedHidden);
        
        W1 += np.matrix(activationHidden).T*errorOutput;

        #deltaOutput = np.multiply(SigmoidDerivative(np.matrix(outputFinal)),error);

        #Delta for hidden layer
        #deltas = 0;

        #for j in range(output):
        #    for w in range(len(W2)):
        #        deltas += np.dot(deltaOutput.T[w],W2.T[w]);
        
        #deltaHidden = SigmoidDerivative(np.matrix(activationHidden)).T*deltas;
        
        #Update Weights
        #W2 += r*np.matrix(error).T*np.multiply(outputFinal,deltaOutput);
        


    return W1,W2;

def NeuralNetwork(epochs,X,Y,r,f,hidden,output):
    W1, W2 = InitializeWeights(f,hidden,output);

    for epoch in range(epochs):
        W1,W2 = Train(X,Y,f,hidden,output,W1,W2,r);

        if(epoch % 10 == 0):
            print epoch;

    return W1,W2;


X, Y, n = Reader.ReadData('data.txt');
W1, W2 = NeuralNetwork(50,X,Y,0.5,4,5,3);
print W1

#newItem = [5.1, 3.5, 1.4, 0.2];
#Predict(newItem,[W1,W2],5,3);

Accuracy(X,Y,[W1,W2],5,3);

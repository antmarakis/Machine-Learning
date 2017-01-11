import numpy as np;
import Reader;

def Predict(item,Weights,hidden,output):
    W1, W2 = Weights;
    
    item = np.append(1,item); #Augment feature vector

    #_Forward Propagation_#
    outputHidden = LayerActivation(hidden,W1,item);
    #Augment hidden activation output
    outputHidden = np.append(1,outputHidden);
    outputFinal = LayerActivation(output,W2,outputHidden);

    #Find max activation in output
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
    #Run training set through network, find overall accuracy
    correct = 0;
    
    for i in range(len(X)):
        x = X[i];
        y = list(Y[i].A1);

        guess = Predict(x,Weights,hidden,output);

        if(y == guess):
            correct += 1;
    
    print correct/float(len(X));

###_Auxiliary Functions_###
def Sigmoid(x):
    return 1 / (1 + np.exp(-x));

def SigmoidDerivative(x):
    return np.multiply(x,1-x);

def InitializeWeights(f,hidden,output):
    #Initialize weights with random values in [-1,1] (including bias)
    #s : length of second layer

    #Augment feature vectors with bias
    f += 1;

    #Initialize second (hidden) layer weights
    W1 = [[np.random.uniform(-1,1) for i in range(f)] for j in range(hidden)];
    W1 = np.matrix(W1);

    #Initialize third (output) layer weights
    W2 = [[np.random.uniform(-1,1) for i in range(hidden+1)] for j in range(output)];
    W2 = np.matrix(W2);
    
    return W1, W2;

###_Core Functions_###
def NeuronActivation(Weight,Input):
    Sum = np.dot(Weight,Input.T);
    
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
        x = np.matrix(np.append(1,x)); #Augment feature vector

        #_Forward Propagation_#
        activationHidden = LayerActivation(hidden,W1,x);
        
        #Augment hidden activation output
        augmentedHidden = np.append(1,activationHidden);
        outputFinal = LayerActivation(output,W2,augmentedHidden);

        #_Backpropagation_#
        #Correction for Output
        aO = np.matrix(outputFinal);
        
        errorOutput = np.matrix(y - outputFinal);
        deltaOutput = np.multiply(errorOutput,SigmoidDerivative(aO));
        
        aA = np.matrix(augmentedHidden);
        D = r*np.multiply(deltaOutput.T,aA);
        
        W2 += D;
        
        #Correction for Hidden
        w = np.delete(W2,[0],axis=1); #remove bias
        
        aH = np.matrix(activationHidden);
        
        errorHidden = np.dot(deltaOutput,w);
        deltaHidden = np.multiply(errorHidden,SigmoidDerivative(aH));
        
        DH = np.multiply(deltaHidden.T,x);
        
        W1 += DH;
    
    return W1,W2;

def NeuralNetwork(epochs,X,Y,r,f,hidden,output):
    W1, W2 = InitializeWeights(f,hidden,output);

    for epoch in range(epochs):
        W1,W2 = Train(X,Y,f,hidden,output,W1,W2,r);

        if(epoch % 10 == 0):
            print epoch;

    return W1,W2;


X, Y, n = Reader.ReadData('data.txt');

f = len(X[0].A1);
h = 9;
o = len(Y[0].A1);

W1, W2 = NeuralNetwork(200,X,Y,0.3,f,h,o);
#newItem = [5.1, 3.5, 1.4, 0.2];
#Predict(newItem,[W1,W2],5,3);

Accuracy(X,Y,[W1,W2],h,o);

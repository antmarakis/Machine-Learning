import numpy as np;
import math;
from numpy.linalg import inv;
from random import shuffle;

###_Pre-Processing_###
def ReadData(fileName):
    f = open(fileName);
    lines = f.read().splitlines();
    f.close();

    items = [];
    classes = [];

    for line in lines:
        line = line.split(','); #Split line on commas
        itemFeatures = []; #Temp list to hold feature values of the item

        for i in range(len(line)-1):
            value = float(line[i]);
            itemFeatures.append(value);

        #Add to classes the known classification for current item
        classes.append(line[-1]);
        #Add item data to items
        items.append(itemFeatures);

    #Map class names to numbers (from 0 to the number of classes)
    classes = map(lambda x: list(set(classes)).index(x), classes);

    X = np.matrix(items); #Convert data to numpy matrix
    Y = BuildY(classes); #Build the Y matrices
    n = len(items)-1; #The number of items

    toShuffle = []; #Temp array to shuffle X and Y at the same time
    
    for i in range(n):
        #Build toShuffle by packing Xi together with Yi
        toShuffle.append((X[i],Y[i]));

    shuffle(toShuffle);
    
    for i in range(n):
        #Unpack toShuffle
        X[i] = toShuffle[i][0];
        Y[i] = toShuffle[i][1];

    return X,Y,n;

def BuildY(Y):
    newY = [];
    #Number of classes is the largest number in Y
    classesNumber = max(Y)+1;

    for i in range(len(Y)):
        #Initialize vector with zeros, set to 1 the class index
        tempVector = [0 for j in range(classesNumber)];
        tempVector[Y[i]] = 1;

        newY.append(tempVector);

    return np.matrix(newY);

def Predict(item,Weights,hidden,output):
    W1, W2 = Weights;
    
    item = np.append(1,item); #Augment feature vector

    #_Forward Propagation_#      
    hiddenActivation = Sigmoid(item*W1);
        
    hiddenActivation = np.append(1,hiddenActivation);
    outputActivation = Sigmoid(hiddenActivation*W2);

    outputActivation = outputActivation.A1;

    m = outputActivation[0];
    index = 0;
    for i in range(1,len(outputActivation)):
        output = outputActivation[i];
        
        if(output > m):
            m = output;
            index = i;

    #Initialize prediction vector to zeros
    y = [0 for i in range(len(outputActivation))];
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
    return np.multiply(x,(1-x));

def InitializeWeights(f,hidden,output):
    #Initialize weights with random values in [0,1] (including bias)
    #s : length of second layer

    #Augment feature vectors with bias
    f += 1;

    #Initialize second (hidden) layer weights
    W1 = [[np.random.uniform(-0.5,0.5) for i in range(f)] for j in range(hidden)];
    W1 = np.matrix(W1);

    #Initialize third (output) layer weights
    W2 = [[np.random.uniform(-0.5,0.5) for i in range(output)] for j in range(hidden+1)];
    W2 = np.matrix(W2);

    return W1, W2;

def Train(X,Y,f,hidden,output,W1,W2,r):
    for i in range(len(X)):
        x = X[i];
        y = Y[i];
        
        x = np.append(1,x);        
        hiddenActivation = Sigmoid(x*W1);
        
        hiddenActivation = np.append(1,hiddenActivation);
        outputActivation = Sigmoid(hiddenActivation*W2);

        error = -(y-outputActivation);
        SigDer = SigmoidDerivative(outputActivation);
        delta3 = np.multiply(error, SigDer);
        dJdW2 = np.dot(np.matrix(hiddenActivation).T,delta3);

        SigDer2 = SigmoidDerivative(hiddenActivation);
        delta2 = np.dot(delta3, W2.T)*np.matrix(SigDer2).T;
        dJdW1 = np.dot(np.matrix(x).T,delta2);

        W1 -= r*dJdW1;
        W2 -= r*dJdW2;
    
    return W1,W2;

def NeuralNetwork(epochs,X,Y,r,f,hidden,output):
    W1, W2 = InitializeWeights(f,hidden,output);

    for epoch in range(epochs):
        W1,W2 = Train(X,Y,f,hidden,output,W1,W2,r);

        if(epoch % 10 == 0):
            print epoch;

    return W1,W2;

hidden = 3;
output = 3;
f = 4;
X, Y, n = ReadData('data.txt');
W1, W2 = NeuralNetwork(50,X,Y,0.5,f,hidden,output);

newItem = [5.1, 3.5, 1.4, 0.2];
print Predict(newItem,[W1,W2],5,3);

#Accuracy(X,Y,[W1,W2],hidden,output);

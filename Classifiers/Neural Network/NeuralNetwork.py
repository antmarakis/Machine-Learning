import numpy as np;
import Reader;


###_Evaluation Functions_###
def Accuracy(X, Y, Weights, layers):
    layers += 1;
    
    #Run training set through network, find overall accuracy
    correct = 0;

    for i in range(len(X)):
        x = X[i];
        y = list(Y[i].A1);

        guess = Predict(x, Weights, layers);

        if(y == guess):
            #Guessed correctly
            correct += 1;

    return correct / float(len(X));

def K_FoldValidation(k, X, Y, f, hiddenLayers, nodes, epochs, r=0.15):
    if (k > len(X)):
        return -1;
    
    correct = 0;  #The number of correct classifications
    total = len(X) * (k - 1);  #The total number of classifications

    l = len(X) / k;  #The length of a fold

    for i in range(k):
        print "\nFold",i;
        
        #Split data set into training and testing
        trainingX = X[i * l:(i + 1) * l];
        trainingY = Y[i * l:(i + 1) * l];

        testX = np.concatenate([X[:i*l],X[(i+1)*l:]]);
        testY = np.concatenate([Y[:i*l],Y[(i+1)*l:]]);

        #Calculate Weights
        weights = NeuralNetwork(epochs,X,Y,f,hiddenLayers,nodes,r);

        #Make predictions for test sets
        for j in range(len(testX)):
            x = testX[j];
            y = list(testY[j].A1);

            guess = Predict(x, weights, hiddenLayers+1);

            if(y == guess):
                #Guessed correctly
                correct += 1;

    return correct / float(total);


###_Auxiliary Functions_###
def Sigmoid(x):
    return 1 / (1 + np.exp(-x));

def SigmoidDerivative(x):
    return np.multiply(x, 1 - x);

def InitializeWeights(f, layers, nodes):
    ##_Initialize weights with random values in [-1,1] (including bias)_##

    #Augment feature vectors with bias
    f += 1;

    #Initialize weights from input to first hidden layer
    inputToHidden = [[np.random.uniform(-1, 1) for i in range(f)] for j in range(nodes[0])];
    inputToHidden = np.matrix(inputToHidden);

    weights = [inputToHidden];
    #Initialize the rest of the weights
    for i in range(1,layers):
        w = [[np.random.uniform(-1, 1) for k in range(nodes[i-1] + 1)] for j in range(nodes[i])];
        w = np.matrix(w);
        weights.append(w);

    return weights;


###_Core Functions_###
def Predict(item, Weights, layers, sigmoid=True):
    item = np.append(1, item);  #Augment feature vector

    ##_Forward Propagation_##
    activations = [item];
    Input = item;
    for i in range(layers):
        activation = np.dot(Input, Weights[i].T);
        if(i < layers-1 or sigmoid):
            #When calculating the output activation, check if
            #we should sigmoid it or not (via the sigmoid var)
            activation = Sigmoid(activation);
        
        activations.append(activation);

        Input = np.append(1, activation); #Augment activation vector
    
    outputFinal = activations[-1].A1;

    #Find max activation in output
    m = outputFinal[0];
    index = 0;
    for i in range(1, len(outputFinal)):
        output = outputFinal[i];

        if (output > m):
            m = output;
            index = i;

    #Initialize prediction vector to zeros
    y = [0 for i in range(len(outputFinal))];
    y[index] = 1;  #Set guessed class to 1

    return y;  #Return prediction vector

def Train(X, Y, r, layers, weights):
    for i in range(len(X)):
        x = X[i];
        y = Y[i].A1;
        x = np.matrix(np.append(1, x));  # Augment feature vector
        
        ##_Forward Propagation_##
        #Each layer receives an input and calculates its output
        #The output of one layer is the input to the next
        #The first input is the first feature vector (the item)
        activations = [x];
        Input = x;
        for i in range(layers):
            activation = Sigmoid(np.dot(Input,weights[i].T));
            activations.append(activation);
            
            Input = np.append(1, activation); #Augment with bias

        ##_Back Propagation_##
        #Find error at output
        #Propagate error backwards through the layers
        #For each layer:
        #a) Calculate delta:
            #Error of next layer * the sigmoid der of current layer activation
        #b) Update weights between current layer and previous layer
            #Multiply delta with activation of previous layer
            #Multiply that with rate
            #Add that to weights of previous layer
        #c) Calculate error for current layer
            #Remove bias from previous-layer weights, get w
            #Multiply delta with w to get error
        outputFinal = activations[-1];
        error = np.matrix(y - outputFinal); #Error at output
        
        for i in range(layers,0,-1):
            currActivation = activations[i];
            
            if(i > 1):
                #Augment previous activation
                prevActivation = np.append(1,activations[i-1]);
            else:
                #First hidden layer, prevActivation is input (without bias)
                prevActivation = activations[i-1];
            
            delta = np.multiply(error, SigmoidDerivative(currActivation));
            weights[i-1] = r * np.multiply(delta.T,prevActivation);

            w = np.delete(weights[i-1], [0], axis=1); #remove bias from weights
            
            error = np.dot(delta,w); #Calculate error for curr layer

    return weights;

def NeuralNetwork(epochs, X, Y, f, hiddenLayers, nodes, r=0.15):
    layers = hiddenLayers + 1; #Total number of layers in network
    weights = InitializeWeights(f, layers, nodes);

    for epoch in range(epochs):
        #Train weights
        weights = Train(X, Y, r, layers, weights);

        if(epoch % 25 == 0):
            print "Epoch ", epoch;

    return weights;


###_Main_###
def main():
    X,Y = Reader.ReadData('data.txt');

    f = len(X[0].A1);
    h1 = 5;
    h2 = 10;
    o = len(Y[0].A1);
    hiddenLayers = 2;
    r = 0.15;
    epochs = 100;

    print K_FoldValidation(5, X, Y, f, hiddenLayers, [h1,h2,o], epochs, r);
    
    #weights = NeuralNetwork(epochs,X,Y,f,hiddenLayers,[h1,h2,o],r);
    #print Accuracy(X, Y, weights, hiddenLayers);

if __name__ == "__main__":
    main();

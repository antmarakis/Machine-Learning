import numpy as np;
import Reader;


###_Auxiliary Functions_###
def Sigmoid(x):
    return 1 / (1 + np.exp(-x));

def SigmoidDerivative(x):
    return np.multiply(x, 1 - x);

def InitializeWeights(f, hidden, output):
    #Initialize weights with random values in [-1,1] (including bias)

    #Augment feature vectors with bias
    f += 1;

    #Initialize second (hidden) layer weights
    W1 = [[np.random.uniform(-1, 1) for i in range(f)] for j in range(hidden)];
    W1 = np.matrix(W1);

    #Initialize third (output) layer weights
    W2 = [[np.random.uniform(-1, 1) for i in range(hidden + 1)] for j in range(output)];
    W2 = np.matrix(W2);

    return W1, W2;


###_Core Functions_###
def Train(X, Y, f, hidden, output, W1, W2, r):
    for i in range(len(X)):
        x = X[i];
        y = Y[i].A1;
        x = np.matrix(np.append(1, x));  # Augment feature vector

        ##_Forward Propagation_##

        #Hidden layer activation
        activationHidden = LayerActivation(hidden, W1, x);

        #Augment hidden activation output
        augmentedHidden = np.append(1, activationHidden);
        outputFinal = LayerActivation(output, W2, augmentedHidden);

        ##_Backpropagation_##

        #_Correction for Output_#

        #Convert output activation to matrix
        aO = np.matrix(outputFinal);

        errorOutput = np.matrix(y - outputFinal);  #Prediction error
        deltaOutput = np.multiply(errorOutput, SigmoidDerivative(aO));

        #Convert hidden activation to matrix
        aA = np.matrix(augmentedHidden);
        D = r*np.multiply(deltaOutput.T, aA);  #Correction for output layer

        W2 += D;  #Correct synapses/weights between hidden-output layers

        #_Correction for Hidden_#
        w = np.delete(W2, [0], axis=1);  #remove bias

        #Convert hidden activation (without bias) to matrix
        aH = np.matrix(activationHidden);

        #Propagate error backwards
        errorHidden = np.dot(deltaOutput, w);
        deltaHidden = np.multiply(errorHidden, SigmoidDerivative(aH));

        DH = r*np.multiply(deltaHidden.T, x);  #Correction for hidden layer

        W1 += DH;  #Correct synapses/weights between input-hidden layers

    return W1, W2;

def NeuronActivation(Weight, Input):
    Sum = np.dot(Weight, Input.T);  #W*x

    #Return sigmoid, turn numpy array to number
    activation = Sigmoid(Sum);
    activation = np.asscalar(activation);

    return activation;

def LayerActivation(Size, Weights, Input):
    layerOutput = [0] * Size;

    for i in range(Size):
        #Build layer activation output via neurons
        w = Weights[i].A1;  #Synapses' Weights
        x = np.matrix(Input);  #Turn input to matrix
        layerOutput[i] = NeuronActivation(w, x);

    return layerOutput;

def NeuralNetwork(epochs, X, Y, r, f, hidden, output):
    W1, W2 = InitializeWeights(f, hidden, output);

    for epoch in range(epochs):
        #Train weights
        W1, W2 = Train(X, Y, f, hidden, output, W1, W2, r);

        if (epoch % 25 == 0):
            print "Epoch ", epoch;

    return W1, W2;


###_Evaluation Functions_###
def Predict(item, Weights, hidden, output):
    W1, W2 = Weights;

    item = np.append(1, item);  # Augment feature vector

    #_Forward Propagation_#
    outputHidden = LayerActivation(hidden, W1, item);
    #Augment hidden activation output
    outputHidden = np.append(1, outputHidden);
    outputFinal = LayerActivation(output, W2, outputHidden);

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

def Accuracy(X, Y, Weights, hidden, output):
    #Run training set through network, find overall accuracy
    correct = 0;

    for i in range(len(X)):
        x = X[i];
        y = list(Y[i].A1);

        guess = Predict(x, Weights, hidden, output);

        if (y == guess):
            correct += 1;

    return correct / float(len(X));

def K_FoldValidation(k, r, X, Y, f, h, o, epochs):
    if (k > len(X)):
        return -1;

    correct = 0;  #The number of correct classifications
    total = len(X) * (k - 1);  #The total number of classifications

    l = len(X) / k;  #The length of a fold

    for i in range(k):
        print "\nNew Fold";
        
        #Split data set into training and testing
        trainingX = X[i * l:(i + 1) * l];
        trainingY = Y[i * l:(i + 1) * l];

        testX = np.concatenate([X[:i*l],X[(i+1)*l:]]);
        testY = np.concatenate([Y[:i*l],Y[(i+1)*l:]]);

        #Calculate Weights
        W1, W2 = NeuralNetwork(epochs, trainingX, trainingY, r, f, h, o);

        #Make predictions for test sets
        for j in range(len(testX)):
            x = testX[j];
            y = list(testY[j].A1);

            guess = Predict(x, [W1, W2], h, o);

            if (y == guess):
                correct += 1;

    return correct / float(total);


###_Main_###
def main():
    X,Y = Reader.ReadData('data.txt');

    f = len(X[0].A1);
    h = 10;
    o = len(Y[0].A1);
    r = 0.15;
    epochs = 200;

    print K_FoldValidation(5, r, X, Y, f, h, o, epochs);

    #W1, W2 = NeuralNetwork(epochs,X,Y,r,f,h,o);
    #print Accuracy(X, Y, [W1,W2], h, o);

if __name__ == "__main__":
    main();

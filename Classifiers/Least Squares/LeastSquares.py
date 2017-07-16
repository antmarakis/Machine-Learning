import numpy as np;
from numpy.linalg import inv;
from random import shuffle;

###_Read Data_###
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

    X, Y = ShuffleArrays(X, Y);

    return X, Y, n;

def ShuffleArrays(A, B):
    toShuffle = []; #Temp array to shuffle X and Y at the same time
    n = len(A);
    
    for i in range(n):
        #Build toShuffle by packing Xi together with Yi
        toShuffle.append((A[i], B[i]));

    shuffle(toShuffle);
    
    for i in range(n):
        #Unpack toShuffle
        A[i] = toShuffle[i][0];
        B[i] = toShuffle[i][1];

    return A,B;

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


###_Core Functions_###
def CalculateWeights(X, Y):
    #Number of attributes
    A = X.shape[1] + 1;
    #Number of classes
    C = Y.shape[1];

    #The sums for Xi*Xi.T and Xi*Yi.T
    XX = [[0.0 for i in range(A)] for j in range(A)];
    XY = [[0.0 for i in range(C)] for j in range(A)];

    for i in range(len(X)):
        x = X[i]; #The ith item vector
        x = np.append(1, x); #Augment item with a 1

        y = Y[i]; #The vector storing the class x is in

        #Calculate outer products of x*x.T and x*y.T
        XX += np.outer(x, x);
        XY += np.outer(x, y);

    XX += 0.001 * np.eye(A); #Avoid XX being non-invertable

    #The weight matrix is the product of XX.T and XY
    weight = np.dot(inv(XX), XY);
    return weight;

def Predict(W,x):
    x = np.append(1, x); #Augment item with a 1

    prediction = np.dot(W.T, x); #List of predictions

    #Find max prediction
    m = prediction[0];
    index = 0;
    for i in range(1,len(prediction)):
        if(prediction[i] > m):
            m = prediction[i];
            index = i;

    #Initialize prediction vector to zeros
    y = [0 for i in range(len(prediction))];
    y[index] = 1; #Set guessed class to 1

    return y; #Return prediction vector


###_Evaluation Functions_###
def K_FoldValidation(k, X, Y):
    if(k > len(X)):
        return -1;

    correct = 0; #The number of correct classifications
    total = len(X)*(k-1); #The total number of classifications

    l = len(X)/k; #The length of a fold

    for i in range(k):
        #Split data set into training and testing
        trainingX = X[i*l:(i+1)*l];
        trainingY = Y[i*l:(i+1)*l];

        testX = np.concatenate([X[:i*l], X[(i+1)*l:]]);
        testY = np.concatenate([Y[:i*l], Y[(i+1)*l:]]);

        W = CalculateWeights(trainingX, trainingY);

        for j in range(len(testX)):
            itemClass = list(testY[j].A1); #The actual classification
            guess = Predict(W, testX[j]); #Make a prediction

            if(guess == itemClass):
                #Guessed correctly
                correct += 1;

    return correct/float(total);

def Evaluate(times, k, X, Y):
    accuracy = 0;
    for t in range(times):
        X, Y = ShuffleArrays(X, Y);
        accuracy += K_FoldValidation(k, X, Y);

    print accuracy/float(times);


###_Main_###
def main():
    X, Y, n = ReadData('data.txt');
    W = CalculateWeights(X, Y);

    Evaluate(100, 5, X, Y);

if __name__ == "__main__":
    main();

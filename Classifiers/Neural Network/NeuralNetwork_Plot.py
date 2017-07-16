import NeuralNetwork as NN;
import Reader;
import numpy as np;
from random import choice;
from matplotlib import pyplot;


def PlotFunction(X, Y, percentage, indexA, indexB):
    n = len(X); #Number of items
    split = int(n*percentage);

    features = len(X[0].A1);
    #Delete all columns but the ones on the given indexes
    for j in range(features):
        if(j == indexA or j == indexB):
            continue;

        X = np.delete(X, j, 1);
    
    testX = X[split:];
    testY = Y[split:];

    #The items will be sorted into classes in this list
    Points = [[] for i in range(len(testY[0].A1))];

    f = 2;
    h1 = 5;
    h2 = 10;
    o = len(Y[0].A1);
    hiddenLayers = 2;
    r = 0.15;
    epochs = 100;

    weights = NN.NeuralNetwork(epochs, X, Y, f, hiddenLayers, [h1,h2,o], r);

    correct = 0;

    #Calculate accuracy
    for i in range(n):
        prediction = NN.Predict(X[i], weights, hiddenLayers+1);
        itemClass = list(Y[i].A1);

        if(prediction == itemClass):
            correct += 1;

        #Find index of class
        index = -1;
        for j in range(len(prediction)):
            if(prediction[j] == 1):
                index = j;
                break;

        Points[index].append(X[i]);

    accuracy = correct/float(n)*100;
    print "Accuracy ",accuracy;

    colors = ['r','b','g','c','m','y'];
    
    for i in range(len(Points)):
        p = Points[i];
        Xa = [];
        Xb = [];

        #Choose color randomly from list, then remove it
        #(to avoid duplicates)
        color = choice(colors);
        colors.remove(color);
        
        for item in p:
            Xa.append(item[:, [0]].item(0));
            Xb.append(item[:, [1]].item(0));

        pyplot.plot(Xa,Xb,'o',color=color);
    
    pyplot.show();


def main():
    X, Y = Reader.ReadData('data.txt');

    PlotFunction(X, Y, 0.7, 2, 3);

main();

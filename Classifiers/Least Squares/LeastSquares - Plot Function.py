import LeastSquares as LS;
import numpy as np;
from random import choice;
from matplotlib import pyplot;


def PlotFunction(X,Y,percentage,indexA,indexB):
    n = len(X)-1; #Number of items
    split = int(n*percentage);

    #Normalize values
    for i in range(n+1):
        X[i] = X[i] / X.max();
    
    testX = X[split:];
    testY = Y[split:];

    #The items will be sorted into classes in this list
    Points = [[] for i in range(testY.shape[1])];

    #Delete all columns but the ones on the given indexes
    for i in range(len(testY[0])):
        if(i == indexA or i == indexB):
            continue;

        testX = np.delete(testX, 0, 1);

    W = LS.CalculateWeights(testX,testY);

    total = len(testX);
    correct = 0;

    #Calculate accuracy
    for i in range(total):
        prediction = LS.Predict(W,testX[i]);
        itemClass = list(testY[i].A1);

        if(prediction == itemClass):
            correct += 1;

        #Find index of class
        index = -1;
        for j in range(len(prediction)):
            if(prediction[j] == 1):
                index = j;
                break;

        Points[index].append(testX[i]);

    accuracy = correct/float(total)*100;
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
        pyplot.plot(W[i],color=color);
    
    pyplot.show();


def main():
    data = LS.ReadData('data2.txt');
    X = data[0];
    Y = data[1];
    n = data[2];

    PlotFunction(X,Y,0.7,2,3);

main();

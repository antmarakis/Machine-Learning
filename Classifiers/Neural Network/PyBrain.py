import Reader;
import Reader_New as Reader2;
import numpy as np;
from pybrain.tools.shortcuts import buildNetwork;
from pybrain.datasets import SupervisedDataSet;
from pybrain.supervised.trainers import BackpropTrainer;


def Predict(net,item):
    output = net.activate(item);

    Max = output[0];
    prediction = 0;
    for i in range(1,len(output)):
        o = output[i];
        if(o > Max):
            Max = o;
            prediction = i;

    #Initialize prediction vector to zeros
    y = [0 for i in range(len(output))];
    y[prediction] = 1; #Set guessed class to 1

    return y; #Return prediction vector

def Accuracy(net,ds):
    correct = 0;
    for inpt, target in ds:
        prediction = Predict(net,inpt);
        
        expectedClass = list(target);
        if(prediction == expectedClass):
            correct += 1;

    print correct/float(n+1);

def FoldValidation(k,X,Y):
    if(k > len(X)):
        return -1;

    correct = 0; #The number of correct classifications
    total = len(X)*(k-1); #The total number of classifications

    l = len(X)/k; #The length of a fold

    for i in range(k):
        #Split data set into training and testing
        trainingX = X[i*l:(i+1)*l];
        trainingY = Y[i*l:(i+1)*l];

        testX = np.concatenate([X[:i*l],X[(i+1)*l:]]);
        testY = np.concatenate([Y[:i*l],Y[(i+1)*l:]]);

        ds = SupervisedDataSet(4,3);

        for j in range(len(trainingX)):
            ds.addSample(trainingX[j],trainingY[j]);

        net = buildNetwork(4,3,3,bias=True);
        trainer = BackpropTrainer(net,ds);
        trainer.trainEpochs(epochs=200);

        for j in range(len(testX)):
            expectedClass = list(testY[j].A1);
            prediction = Predict(net,testX[j].A1);
            
            if(prediction == expectedClass):
                correct += 1;

    print correct/float(total);

def main():
    ds = SupervisedDataSet(4,3);

    for i in range(len(X)):
        ds.addSample(X[i],Y[i]);

    net = buildNetwork(4,9,3,bias=True);
    trainer = BackpropTrainer(net,ds);
    trainer.trainEpochs(epochs=200);

    Accuracy(net,ds);


X,Y,classNames = Reader2.ReadData('data.txt');
n = len(X)-1;
#FoldValidation(5,X,Y);
main();

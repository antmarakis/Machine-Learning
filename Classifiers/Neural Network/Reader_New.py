import numpy as np;
import math;
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

    X = np.matrix(items); #Convert data to numpy matrix
    n = len(items)-1; #The number of items
    classNames = list(set(classes));
    #Map class names to numbers (from 0 to the number of classes)
    classes = map(lambda x: list(set(classes)).index(x), classes);
    Y = BuildY(classes); #Build the Y matrices

    toShuffle = []; #Temp array to shuffle X and Y at the same time
    
    for i in range(n):
        #Build toShuffle by packing Xi together with Yi
        toShuffle.append((X[i],Y[i]));

    shuffle(toShuffle);
    
    for i in range(n):
        #Unpack toShuffle
        X[i] = toShuffle[i][0];
        Y[i] = toShuffle[i][1];

    return X,Y,classNames;

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

ReadData('data.txt');

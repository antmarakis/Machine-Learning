import kMeans;
import numpy as np;
from random import choice;
from matplotlib import pyplot;

def CutToTwoFeatures(items,indexA,indexB):
    n = len(items);
    X = [];
    for i in range(n):
        item = items[i];
        newItem = [item[indexA],item[indexB]];
        X.append(newItem);

    return X;
        

def PlotClusters(clusters):
    n = len(clusters);
    #Cut down the items to two dimension and store to X
    X = [[] for i in range(n)];

    for i in range(n):
        cluster = clusters[i];
        for item in cluster:
            X[i].append(item);

    colors = ['r','b','g','c','m','y'];

    for x in X:
        #Choose color randomly from list, then remove it
        #(to avoid duplicates)
        c = choice(colors);
        colors.remove(c);

        Xa = [];
        Xb = [];

        for item in x:
            Xa.append(item[0]);
            Xb.append(item[1]);

        pyplot.plot(Xa,Xb,'o',color=c);

    pyplot.show();
        

def main():
    items = kMeans.ReadData('data.txt');
    items = CutToTwoFeatures(items,2,3);
    
    k = 3;
    means = kMeans.CalculateMeans(k,items);
    clusters = kMeans.FindClusters(means,items);
    
    PlotClusters(clusters);

main();

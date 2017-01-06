import kMeans;
import numpy as np;
from random import choice;
from matplotlib import pyplot;


def PlotClusters(clusters,indexA,indexB):
    n = len(clusters);
    #Cut down the items to two dimension and store to X
    X = [[] for i in range(n)];

    for i in range(n):
        cluster = clusters[i];
        for item in cluster:
            newX = [item[indexA],item[indexB]];

            X[i].append(newX);

    colors = ['r','b','g','c','m','y'];

    for x in X:
        #Choose color randomly from list, then remove it
        #(to avoid duplicates)
        color = choice(colors);
        colors.remove(color);

        Xa = [];
        Xb = [];

        for item in x:
            Xa.append(item[0]);
            Xb.append(item[1]);

        pyplot.plot(Xa,Xb,'o',color=color);

    pyplot.show();
        

def main(k):
    data = kMeans.ReadData('data.txt');
    items = data[0];
    colMinima = data[1];
    colMaxima = data[2];

    #Find means and clusters
    means = kMeans.CalculategMeans(k,items,colMinima,colMaxima);
    clusters = kMeans.FindClusters(means,items);
    
    PlotClusters(clusters,2,3);

main(3);

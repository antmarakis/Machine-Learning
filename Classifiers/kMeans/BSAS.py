import math; #For pow, sqrt and ceil
import sys; #For maxint
import numpy as np;
from random import shuffle;


###_Auxiliary Function_###
def FindMinMax(items):
    #Find min and max distances between all items
    n = len(items);
    minimum = sys.maxint;
    maximum = -sys.maxint - 1;
    
    for i in range(n):
        for j in range(i+1,n):
            dis = EuclideanDistance(items[i],items[j]);

            if(dis < minimum):
                minimum = dis;

            if(dis > maximum):
                maximum = dis;

    return minimum,maximum;

def FindMax(Dict):
    maximum = -1;
    key = -1;

    for k in Dict.keys():
        if(Dict[k] > maximum):
            maximum = Dict[k];
            key = k;

    return key;

def FindMostFrequent(Dict):
    frequency = {};

    for k in Dict.keys():
        if(Dict[k] not in frequency):
            frequency[Dict[k]] = 1;
        else:
            frequency[Dict[k]] += 1;

    return FindMax(frequency);

def EuclideanDistance(x,y):
    S = 0; #The sum of the squared differences of the elements
    for i in range(len(x)):
        S += math.pow(x[i]-y[i],2);

    return math.sqrt(S); #The square root of the sum

def UpdateMean(n,mean,item):
    for i in range(len(mean)):
        m = mean[i];
        m = (m*(n-1)+item[i])/float(n);
        mean[i] = round(m,3);
    
    return mean;

###_Core Functions_###
def Classify(means,item):
    #Item is classified to the mean with minimum distance
    minimum = sys.maxint;
    index = -1;

    for i in range(len(means)):
        #Find distance from item to mean
        dis = EuclideanDistance(item,means[i]);
        dis = round(dis,3);

        #dis is shorter than current minimum
        #Update minimum and index of minimum
        if(dis < minimum):
            minimum = dis;
            index = i;
    
    return index,dis;

def CalculateMeans(items,theta):
    #Initialize first mean at first item
    means = [items[0]];
    
    #Initialize clusters, the array to hold
    #the number of items in a cluster
    clusters = [1];
    
    for i in range(1,len(items)):
        #Iterate through all items (except the first)
        item = items[i];
        
        #Pick cluster closer to item
        index, dis = Classify(means,item);

        if(dis > theta):
            #Distance is larger than threshold
            #Create new cluster
            means.append(item);
            clusters.append(1);
        else:
            clusters[index] += 1;
            means[index] = UpdateMean(clusters[index],means[index],item);

    return len(means);

def CalculateK(items,s):
    #s : Number of runs of BSAS
    #q : Maximum number of classes
    #The min and max distances
    temp = FindMinMax(items);
    minimum = temp[0];
    maximum = temp[1];

    #The range of values for theta/threshold
    #tRange = np.arange(minimum,maximum,0.1);
    tRange = np.arange(3.1,3.6,0.01);

    thetaMeans = {};
    
    for theta in tRange:
        #Count how many times each number of classes appears
        m = {};
        
        for i in range(s):
            shuffle(items);

            #Calculate number of classes
            n = CalculateMeans(items,theta);

            #Increment class number counter
            if(n not in m.keys()):
                m[n] = 1;
            else:
                m[n] += 1;

        thetaMeans[round(theta,3)] = FindMax(m);

    print sorted(thetaMeans.iteritems())
    return FindMostFrequent(thetaMeans);

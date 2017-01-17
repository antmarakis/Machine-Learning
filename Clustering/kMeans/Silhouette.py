from sklearn.metrics import silhouette_score
import kMeans;


def FindMax(Dict):
    #Find key with largest value
    maximum = -1;
    key = -1;

    for k in Dict.keys():
        if(Dict[k] > maximum):
            maximum = Dict[k];
            key = k;

    return key;

def CalculateK(items,s):
    clusterRange = [3,4,5,6,7,8,9]; #The range of k
    clusterCount = {};
    maximum = -100;
    k = -1;

    for i in range(s):
        #Run algorithm s times
        for n in clusterRange:
            #Iterate through possible cluster sizes
            means = kMeans.CalculateMeans(n,items,5);
            classifications = []; #Matrix of classifications
            
            for item in items:
                #Predict cluster classification for each item
                prediction = kMeans.Classify(means,item);
                classifications.append(prediction);

            #Calculate silhouetting score
            sScore = silhouette_score(items,classifications);

            #Find maximum silhouetting score
            if(sScore > maximum):
                maximum = sScore;
                k = n;

        #Calculate number of occurences of k
        if(k not in clusterCount):
            clusterCount[k] = 1;
        else:
            clusterCount[k] += 1;

    #Return the most frequent number of clusters
    return FindMax(clusterCount);

import math; #For pow and sqrt
from random import shuffle;


###_Read Data_###
def ReadData(fileName):
    #Read the file, splitting by lines
    f = open(fileName,'r');
    lines = f.read().splitlines();
    f.close();

    #Split the first line by commas, remove the last element
    #and save the length of the rest.
    featuresNumber = len(lines[0].split(','));

    items = [];
    classes = [];
    features = lines[0].split(',')[:-1];

    for i in range(1,len(lines)):
        line = lines[i].split(',');

        if(line[-1] not in classes):
            classes.append(line[-1]);

        itemFeatures = {"Class" : line[-1], "Bias" : 1};

        for j in range(len(features)):
            #Iterate through the features
            f = features[j]; #Get the feature at index j
            v = float(line[j]); #Convert feature value to float

            itemFeatures[f] = v;
    
        items.append(itemFeatures);

    shuffle(items);

    return items,classes,features;


###_Evaluation Functions_###
def K_FoldValidation(K,Items,rate,epochs,classes,features):
    if(K > len(Items)):
        return -1;

    correct = 0; #The number of correct classifications
    total = len(Items)*(K-1); #The total number of classifications

    l = len(Items)/K; #The length of a fold

    for i in range(K):
        #Split data set into training and testing
        trainingSet = Items[i*l:(i+1)*l];
        testSet = Items[:i*l] + Items[(i+1)*l:];

        weights = CalculateWeights(trainingSet,rate,epochs,classes,features);

        for item in testSet:
            itemClass = item["Class"];

            itemFeatures = {};

            for key in item:
                if(key != "Class"):
                    #If key isn't "Class", add it to itemFeatures
                    itemFeatures[key] = item[key];
          
            guess = Perceptron(itemFeatures,weights);

            if(guess == itemClass):
                #Guessed correctly
                correct += 1;

    return correct/float(total);

def Evaluate(times,K,Items,rate,epochs,classes,features):
    accuracy = 0;
    for t in range(times):
        shuffle(Items);
        accuracy += K_FoldValidation(K,Items,rate,epochs,classes,features);

    print accuracy/float(times);


###_Auxiliary Functions_###
def AddDictionaries(d1,d2,rate):
    d3 = {};
    for i in d1:
        d3[i] = d1[i] + rate*d2[i];

    return d3;

def SubDictionaries(d1,d2,rate):
    d3 = {};
    for i in d1:
        d3[i] = d1[i] - rate*d2[i];

    return d3;


###_Core Functions_###
def CalculateConfidence(item,weight):
    #Add the product of the weight and item values for each feature
    confidence = 0;

    for k in weight:
        confidence += weight[k]*item[k];

    return confidence;

def CalculateWeights(trainingSet,rate,epochs,classes,features):
    #Initialize weights at 0
    weights = {};

    #Initialize weights dictionary. Weights is divided in classes.
    #Each class has its own dictionary, which is numerical values/weights
    #for the features.
    for c in classes:
        weights[c] = {"Bias":0};
        for f in features:
            weights[c][f] = 0;

    for epoch in range(epochs):
        for item in trainingSet:
            #Iterate through trainingSet
            #Guess where item belongs
            y = -1;
            guess = "";
            for w in weights:
                confidence = CalculateConfidence(item,weights[w]);

                if(confidence > y):
                    y = confidence;
                    guess = w;

            correct = item["Class"];
            if(correct != guess):
                weights[guess] = SubDictionaries(weights[guess],item,rate);
                weights[correct] = AddDictionaries(weights[correct],item,rate);

    return weights;

def Perceptron(item,weights):
    item["Bias"] = 1; #Augment item vector with bias
    m = -1; #Hold the maximum
    classification = ""; #Hold the classification

    #Calculate chance of item being in each class,
    #pick the maximum
    for w in weights:
        #Multiply the item vector with the class weights vector
        guess = CalculateConfidence(item,weights[w]);
        if(guess > m):
            #Our guess is better than our current best guess,
            #update max and classification
            m = guess;
            classification = w;

    return classification;


###_Main_###
def main():
    data = ReadData('data.txt');

    items = data[0];
    classes = data[1];
    features = data[2];

    lRate = 0.1;
    epochs = 50;
    weights = CalculateWeights(items,lRate,epochs,classes,features);

    item = {'PW' : 1.4, 'PL' : 4.7, 'SW' : 3.2, 'SL' : 7.0};
    print Perceptron(item,weights);

    #Evaluate(100,5,items,lRate,epochs,classes,features);

if __name__ == "__main__":
    main();

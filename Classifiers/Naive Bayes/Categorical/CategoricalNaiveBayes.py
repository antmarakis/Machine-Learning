import _DataReader as DataReader;


def Classifier(Evidence):
    #The string of evidence, so that we can save it in P.
    evidence = '';

    #Check if all evidence is also in Features
    for e in Evidence:
        if e not in Features:
            #A given evidence does not belong in Features. Abort.
            print "Evidence list is erroneous"
            return;

        #Build the evidence string
        evidence += e + ', ';

    evidence = evidence[:-2]; #remove the last two chars, as they are ', '

    m = -1.0; #Hold the max
    classification = ''; #Hold the classification

    #We need to find P(c|evidence). The equation (from Bayes) is:
    #P(c|evidence) = P(evidence|c)*P(c)/P(evidence)
    #Because this Bayes classifier is naive, the features in evidence are
    #independent. Therefore, the above equation is simplified to:
    #P(c|evidence) = P(evidence1|c)*P(evidence2|c)*...*P(evidenceN|c) * P(c)
    #divided by P(evidence1)*P(evidence2)*...*P(evidenceN)

    #Calculate the probability of all classes for given evidence/features
    #using the Bayes equation. Pick the highest.
    for c in Classes:
        P[c + '|' + evidence] = P[c]; #Start from the prior probability
        
        for e in Evidence:
            #Multipy by the conditional prob and divide by the feature prob
            P[c + '|' + evidence] *= P[e + '|' + c] / P[e];

        #Find the max
        if(P[c + '|' + evidence] > m):
            #P(c|evidence) is the max so far; update m and classification
            m = P[c + '|' + evidence];
            classification = c;

    #With the evidence, the item belongs to classifaction with a prob of m
    print classification, m;


#Read data from file
data = DataReader.Read('data4.txt');
Classes = data[0];
Features = data[1];
P = data[2];

#Run classifier with the evidence list
Classifier(['Tall','Slim']);

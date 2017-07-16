import _DataReader as DataReader;
import math;


def Gaussian(mean, stDev, x):
    g = 1/(math.sqrt(2*math.pi)*stDev) * math.e**(-0.5*(float(x-mean)/stDev)**2);
    return g;

def Classifier(Evidence):
    #The string of evidence, so that we can save it in P.
    evidence = '';

    #Check if all evidence is also in Features
    for e in Evidence:
        eF = e[0]; #The feature in evidence e
        eV = e[1]; #The value in evidence e
        if eF not in Features:
            #A given evidence does not belong in Features. Abort.
            print "Evidence list is erroneous";
            return;

        #Build the evidence string
        evidence += eF + " = " + str(eV) + ', ';

    evidence = evidence[:-2]; #remove the last two chars, as they are ', '

    m = -1.0; #Hold the max
    classification = ''; #Hold the classification

    #We need to find P(c|evidence). The equation (from Bayes) is:
    #P(c|evidence) = P(evidence|c)*P(c)/P(evidence)
    #Because this Bayes classifier is naive, the features in evidence are
    #independent. Therefore, the above equation is simplified to:
    #P(c|evidence) = P(evidence1|c)*P(evidence2|c)*...*P(evidenceN|c) * P(c).
    #We do not need to calculate P(evidence) as it is the same for all
    #classes.

    #We know the individual probability P(c) but we do not know the
    #probability of the conditional probabilities P(evidenceX|c).
    #We calculate those using the Gaussian distribution formula.
    #Instead of Gaussian we can use any other distribution, if it is known.

    #The parameters are the mean, the standard deviation and the value of the evidence.
    #We have the value from the evidence, eV.
    #The mean is the class mean for the feature, Classes[c][eF]["Mean"].
    #The stDev is the class stDev for the feature, Classes[c][eF]["StDev"].

    #We input those to the Gaussian formula and we receive the output.

    #Calculate the probability of all classes for given evidence/features
    #using the Bayes equation. Pick the highest.
    for c in Classes:
        P[c + '|' + evidence] = P[c]; #Start from the prior probability
        
        for e in Evidence:
            eF = e[0]; #The feature in evidence e
            eV = e[1]; #The value in evidence e
            #Multipy by the conditional prob
            mean = Classes[c][eF]["Mean"]; #mean
            stDev = Classes[c][eF]["StDev"]; #standard deviation
            P[c + '|' + evidence] *= Gaussian(mean,stDev,eV);

        if(P[c + '|' + evidence] > m):
            #P(c|evidence) is the max so far; update m and classification
            m = P[c + '|' + evidence];
            classification = c;

    #With the evidence, the item belongs to classifaction with a prob of m
    print classification, m;


#Read data from file
Classes, Features, P, n = DataReader.Read('data.txt'); #Returns a tuple

#Run classifier with the evidence list
Classifier((('Height', 170), ('Weight', 65)));

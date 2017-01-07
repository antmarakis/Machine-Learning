import math;

def Read(fileName):
    Classes = {}; #Class dictionary
    Features = []; #Features list

    #Read data from input file, split the lines
    f = open(fileName,'r');
    lines = f.read().splitlines();
    f.close();

    n = len(lines)-1; #The size of the data set

    #Extract the features
    features = lines[:1][0]; #The first line of input, taking it as a string.
    features = features.split(' ')[1:]; #Split first line by spaces
    l = len(features); #The number of features

    #Extract the class data
    classes = lines[1:]; #Remove the first line (the features)

    for f in features:
        #For every string in the first line, add a new item to Features, plus
        #its complement.
        Features.append(f);

    #Construct Classes table#
    #a) Find means
    for c in classes:
        #Split current line (item) by spaces
        #The first element holds the name of the class
        #The rest show whether the item has a certain feature
        c = c.split(' ');

        if(c[0] not in Classes):
            #The item class has not been added to Classes. Add it now.
            Classes[c[0]] = {"Total":0}; #Set the total of the class to 0.
            for f in Features:
                #Add to the class dictionary (table) all the features' mean
                #and standard deviation
                Classes[c[0]][f] = {"Mean":0, "StDev":0};

        #Increment the total items in the item class
        Classes[c[0]]["Total"] += 1;

        #Calculate the mean of classes' features
        for i in range(1,l+1):
            t = Classes[c[0]]["Total"]; #Pass the total
            f = Classes[c[0]][features[i-1]]["Mean"]; #The current average

            Classes[c[0]][features[i-1]]["Mean"] = (f*(t-1)+float(c[i]))/t;

    #b) Find Standard Deviations
    #StDev : Square Root of Variance
    #Variance : Average of squared difference from mean
    
    values = {};
    for k in Classes.keys():
        #We will save the variances in here, building them up as we go
        #so that we can use them at the end.
        values[k] = {};
        for f in Features:
            values[k][f] = 0;
    
    for c in classes:
        #To find the standard deviations, we first need to know the means.
        #That's why we run this loop after the first loop.
        c  = c.split(' ');

        for i in range(1,l+1):
            #From the current value, substract the mean.
            #Divide by the total (-1) to get the average.
            #We are using the total minus one because we do not want the
            #population deviation, but the standard deviation.
            v = math.pow(int(c[i]) - Classes[c[0]][features[i-1]]["Mean"],2);
            values[c[0]][Features[i-1]] += v/(Classes[c[0]]["Total"]-1);

        for k in Classes.keys():
            #Calculate StDev for the features of each class, using values
            for i in range(1,l+1):
                #The Standard Deviation is the square root of the variance
                s = math.sqrt(values[k][Features[i-1]]);
                
                Classes[k][features[i-1]]["StDev"] = s;


    #Calculate the various probabilities
    P = {}; #Probability dictionary. Holds the various probabilities

    #Calculate the prior probabilities of the classes
    for c in Classes:
        P[c] = Classes[c]["Total"]/float(n);
                
    
    return (Classes, Features, P, n);

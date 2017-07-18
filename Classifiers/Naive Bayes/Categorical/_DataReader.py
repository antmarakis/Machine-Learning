def Read(fileName):
    Classes = {};
    Features = {};

    #Read data from input file, split the lines
    f = open(fileName,'r');
    lines = f.read().splitlines();
    f.close();

    n = len(lines)-1; #The size of the data set

    #Extract the features
    features = lines[:1][0]; #The first line of input, taking it as a string.
    features = features.split(' ')[1:]; #Split first line by spaces
    l = len(features);

    #Extract the class data
    classes = lines[1:]; #Remove the first line

    for f in features:
        #For every string in the first line, add a new item to Features,
        #plus its complement.
        Features[f] = 0;
        Features["Not " + f] = 0;

    #Construct Classes table
    for c in classes:
        #Split current line (item) by spaces
        #The first element holds the name of the class
        #The rest show whether the item has a certain feature
        c = c.split(' ');

        if(c[0] not in Classes):
            #The item class has not been added to Classes. Add it now.
            Classes[c[0]] = {"Total":0}; #Set the total of the class to 0.
            for f in Features:
                #Add to the class dictionary (table) all the features, set to 0.
                Classes[c[0]][f] = 0;

        #Increment the total items in the item class
        Classes[c[0]]["Total"] += 1;

        for i in range(1,l):
            if(c[i] == 'True'):
                #The item has the feature in the ith index in the item list, c
                #The ith index in c corresponds with the i-1 index in features
                feature = features[i-1]; #Save it in feature
            elif(c[i] == 'False'):
                #The item doesn't have the feature in the item list
                #Instead, it has the "Not Feature", the complement of the feature
                feature = "Not " + features[i-1]; #Save complement in feature

            Features[feature] += 1; #Increment feature counter

            if(feature not in Classes[c[0]]):
                #The feature has not been added to the class dictionary.
                #Add feature to the item class.
                Classes[c[0]][feature] = 1;
            else:
                #The feature exists in the class dictionary.
                #Increment the feature counter in the item class.
                Classes[c[0]][feature] += 1;


    #Calculate the various probabilities
    P = {}; #Probability dictionary. Holds the various probabilities

    #Calculate the prior probabilities of the classes
    for c in Classes:
        P[c] = Classes[c]["Total"]/float(n);

    #Calculate the prior probabilities of the features
    for f in Features:
        P[f] = Features[f]/float(n);

    #Calculate the conditional probabilities
    for c in Classes:
        for f in Features:
            P[f + '|' + c] = Classes[c][f]/float(Classes[c]["Total"]);

    return (Classes, Features, P);

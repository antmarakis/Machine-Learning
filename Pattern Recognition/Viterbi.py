def Viterbi(transitionProb,startProb,emissionProb,states,observations,n):
    V = [];
    firstObs = observations[0];
    
    #Calculate first column
    v = {}; #Temp dict to hold state data
    for s in states:
        #Add dummy value to PrevState key
        v[s] = {"PrevState":"Start"};
        #The probability of state appearing first is its prob
        #of appearing first in general times the prob of
        #the first observation being emitted from that state.
        v[s]["Prob"] = startProb[s]*emissionProb[s][firstObs];
    V.append(v.copy());
    
    for i in range(1,n):
        obs = observations[i]; #Current observation
        
        v = {}; #Temp dict
        for s in states:
            #Find max prob of states for current observation
            maxProb = -1;
            state = ""; #Hold previous state where maxProb occured
            
            emission = emissionProb[s][obs];
            for prevState in states:
                #Find max from previous column
                prevProb = V[i-1][prevState]["Prob"];
                prob = prevProb*transitionProb[prevState][s]*emission;
                
                if(prob > maxProb):
                    maxProb = prob;
                    state = prevState;
            
            v[s] = {};
            v[s]["Prob"] = maxProb;
            v[s]["PrevState"] = state;
        
        V.append(v.copy());
    
    #Find max probability
    maxProb = -1;
    state = "";
    for s in states:
        if(V[-1][s]["Prob"] > maxProb):
            maxProb = V[-1][s]["Prob"];
            state = s;
    
    #Find sequence by moving back from the final state
    sequence = [state];
    for i in range(n-2,-1,-1):
        prevState = V[i+1][state]["PrevState"]
        sequence.insert(0,prevState); #Insert prevState to start of seq
        state = prevState;
    
    print sequence;
    print maxProb;


def Initialization1():
    ## Example data from Wikipedia ##
    transitionProb = {
        "Healthy": {"Healthy":0.7,"Fever":0.3},
        "Fever" : {"Healthy":0.4,"Fever":0.6}
    };

    emissionProb = {
        "Healthy":{"Normal":0.5,"Cold":0.4,"Dizzy":0.1},
        "Fever":{"Normal":0.1,"Cold":0.3,"Dizzy":0.6}
    };

    startProb = {"Healthy":0.6,"Fever":0.4};

    states = ["Healthy","Fever"];
    observations = ["Normal","Cold","Dizzy"];
    n = len(observations);
    
    return transitionProb,startProb,emissionProb,states,observations,n;

def Initialization2():
    transitionProb = {
        "Rain": {"Rain": 0.5, "Sun": 0.1, "Cloud": 0.4},
        "Cloud": {"Rain": 0.3, "Sun": 0.3, "Cloud": 0.4},
        "Sun": {"Rain": 0.1, "Sun": 0.5, "Cloud": 0.4},
    };

    emissionProb = {
        "Rain": {"Walk": 0.1, "Clean": 0.3, "Study": 0.5, "Shop": 0.1},
        "Cloud": {"Walk": 0.3, "Clean": 0.2, "Study": 0.3, "Shop": 0.2},
        "Sun": {"Walk": 0.5, "Clean": 0.1, "Study": 0.1, "Shop": 0.3},
    };

    startProb = {"Rain": 0.2, "Cloud": 0.3, "Sun": 0.4};

    states = ["Rain", "Cloud","Sun"];
    observations = ["Walk", "Walk", "Shop", "Walk", "Study", "Study"];
    n = len(observations);
    
    return transitionProb,startProb,emissionProb,states,observations,n;


transitionProb,startProb,emissionProb,states,observations,n = Initialization2();

Viterbi(transitionProb,startProb,emissionProb,states,observations,n);

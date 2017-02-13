def EditDistance(a,b):
    len1 = len(a);
    len2 = len(b);
    
    D = [[0 for j in range(len2+1)] for i in range(len1+1)];
    
    for i in range(1,len1+1):
        D[i][0] = i;
    
    for j in range(1,len2+1):
        D[0][j] = j;
    
    for i in range(1,len1+1):
        for j in range(1,len2+1):
            #If chars are the same, added cost is 0
            same = 0;
            if(a[i-1] != b[j-1]):
                #Otherwise it is 1 (for replace)
                same = 1;
            
            c1 = D[i-1][j-1] + same; #diagonally
            c2 = D[i-1][j] + 1; #from below
            c3 = D[i][j-1] + 1; #from left
            
            D[i][j] = min(c1,c2,c3);
    
    print D;
    return D[-1][-1];


word1 = "kitten";
word2 = "cat";

print EditDistance(word1,word2);
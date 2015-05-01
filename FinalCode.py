#References
#No of Hidden Layers and Nodes : http://goo.gl/OdBRY2
#No of Hidden Layers and Nodes : http://goo.gl/Nb4GZp

#Header Files and Imports
import random
import math
import numpy as np
#Reading the Data from txt files
#ref : https://docs.python.org/2/tutorial/inputoutput.html#reading-and-writing-files
def prepareInput(act_file,inact_file):
    active_TD = open(act_file,'r')
    inactive_TD = open(inact_file,'r')

    #initialize Data Matrix to store training Data
    DataMatrix = []
    OutputMatrix = []
    #Truncate the First row to remove the attribute labels
    Garbage = active_TD.readline()
    Garbage = inactive_TD.readline()

    while 1:
        curr_row_act = active_TD.readline()
        curr_row_inact = inactive_TD.readline()
        if curr_row_inact:
            split_row = [float(x) for x in curr_row_inact.split()]
            #Remove the first column. We dont need ID of the compound here
            split_row = split_row[1:]
            DataMatrix.append(split_row)
            OutputMatrix.append([0])
        if curr_row_act:
            split_row = [float(x) for x in curr_row_act.split()]
            #Remove the first column. We dont need ID of the compound here            
            split_row = split_row[1:]
            DataMatrix.append(split_row)
            OutputMatrix.append([1])
        if not ( curr_row_act or curr_row_inact):
            break
    return (DataMatrix, OutputMatrix)

#Intialize the weights to some random value before starting the training of the model
def initializeRandomWeights(row,col,a,b):
    weights= []
    for i in range(0,row):
        r = []
        for j in range(0, col):
            r.append(random.uniform(a, b))
        weights.append(r)
    return weights    

#Matrix Multiplication
def MatMul(X,Y):
    if len(X[0]) != len(Y):
        print("Matrix Dimensions poblem : " , len(X[0]) , len(Y))
        exit()
    Z=[[0]*len(Y[0])]*len(X)
    for i in range(len(X)):
        for j in range(len(Y[0])):
            for k in range(len(Y)):
                    Z[i][j] += X[i][k] * Y[k][j]
    return Z


# our sigmoid function, standard 1/(1+e^-x)
def sigmoid(x):
    return 1/(1+(math.e**(-1*x)))

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def forwardpropagate(XL1,Y,WL1,WL2):
    XL2 = np.dot(XL1,WL1)
    ZL2 = XL2
    SXL2 = []
    for i in range(len(XL2)):
        r = []
        for j in range(len(XL2[0])):
            r.append(sigmoid(XL2[i][j]))
        SXL2.append(r)    
    Output = np.dot(SXL2,WL2)
    ZL3 = Output
    SOutput = [[0]*len(Output[0])]*len(Output)
    for i in range(len(Output)):
        for j in range(len(Output[0])):
            SOutput[i][j] = sigmoid(Output[i][j]) 
    return (SOutput,XL2,ZL2,ZL3)


def transpose(X):
    tX = [[0]*len(X)]*len(X[0])
    for i in range(len(X)):
        for j in range(len(X[0])):
            tX[j][i] = X[i][j]
    return tX
             
#Compute the error
def cost(actual,expected,inputCount):
    JVal = 0
    for i in range(len(actual)):
        for j in range(len(actual[0])):
            try:
                JVal += (actual[i][j]*math.log(expected[i][j]) + (1-actual[i][j])*math.log(1-expected[i][j]))
            except:
                pass    
    return ((JVal*1.0)/inputCount)  

#Normalize the Input Data - Important
def featureScaling(X):
    cnt=0
    row = len(X)
    col = len(X[0])
    mean = [0]*col
    sd = [0]*col
    maxval = [-100000000000]*col
    minval = [1000000000000]*col
    for i in range(col):
        for j in range(row):
            mean[i]+=X[j][i]
            maxval[i] = max(maxval[i],X[j][i])
            minval[i] = min(minval[i],X[j][i])
        sd[i] = maxval[i]-minval[i]
        mean[i] = (mean[i]*1.0)/row
    for i in range(row):
        for j in range(col):    
            try:
                X[i][j] = ((X[i][j]-minval[j])*1.0)/(maxval[j]-minval[j])
            except:
                X[i][j] = 0
                pass    
    return X


#Adjust the weights by back propagating the errors in each layer        
def backpropagate(expected,actual,XL1,XL2,Z2,m):
    dL3  = []
    for i in range(len(expected)):
        dL3.append([(expected[i][0] - actual[i][0])])
    print("Calculaing Delta-2")    
    temp_dL2  = np.dot(dL3,transpose(WL2))
    dL2 = []
    for i in range(len(temp_dL2)):
        r = []
        for j in range(len(temp_dL2[0])):
            r.append(temp_dL2[i][j]*dsigmoid(Z2[i][j]))
        dL2.append(r)    
    print("Calculaing Delta-Caps")        
    DL2 = np.dot(transpose(XL2),dL3)
    DL1 = np.dot(transpose(XL1),dL2)
    W1GRAD=[]
    W2GRAD=[]
    for i in range(len(DL1)):
        r = []
        for j in range(len(DL1[0])):
            r.append((1.0/m)*DL1[i][j])
        W1GRAD.append(r)    
    for i in range(len(DL2)):
        r=[]
        for j in range(len(DL2[0])):
            r.append((1.0/m)*DL2[i][j])
        W2GRAD.append(r)    
    return (W1GRAD,W2GRAD)        

#Formulaes for Data Metrics

def recall(TP,FN):
    return (TP*1.0)/(TP+FN)

def precision(TP,FP):
    return (TP*1.0)/(TP+FP)

def accuracy(TP,FP,TN,FN):
    return ((TP+TN)*1.0)/(TP+FP+TN+FN)

def F_score(p,r):
    return 2*p*r/(p+r)

def metrics(expected,actual):
    normalized_expected = []
    for x in expected:
        if x[0]>=0.7:
            normalized_expected.append([1])
        else:
            normalized_expected.append([0])
    TP=TN=FP=FN=0
    for i in range(len(actual)):
        if actual[i][0] == 1 and normalized_expected[i][0] == 0:
            FN+=1
        if actual[i][0] == 0 and normalized_expected[i][0] == 0:
            TN+=1    
        if actual[i][0] == 1 and normalized_expected[i][0] == 1:
            TP+=1
        if actual[i][0] == 0 and normalized_expected[i][0] == 1:
            FP+=1
    print("False Positives  : ",FP)
    print("True Positives  : ",TP)
    print("False Negitives  : ",FN)
    print("True Negitives  : ",TN)
    print("Accuracy  : ",accuracy(TP,FP,TN,FN))
    print("Recall  : ",recall(TP,FN))
    print("Precision  : ",precision(TP,FP))
    print("F-Score  : ",F_score(precision(TP,FP),recall(TP,FN))) 

    
#Input Raw Data and get Preprocessed Data
TD, TR = prepareInput('Datasets/1332/1332- active.txt','Datasets/1332/1332- inactive.txt')

#Intialize the weights for both layers
rows = len(TD[0])
cols = len(TD)/2
WL1 = initializeRandomWeights(rows,cols,-0.01,0.01)
WL2 = initializeRandomWeights(cols,1,-0.01,0.01)
prev_cost = 1111111111

print("Normalizing Data")
TD = featureScaling(TD)
while 1:
    print("Forward propagating")
    Expected_Output , A2 , Z2 , Z3 = forwardpropagate(TD,TR,WL1,WL2)
    curr_cost = cost(TR,Expected_Output,len(TR))
    if abs(curr_cost - prev_cost) <= 0.0001:
        break
    print(curr_cost)    
    print(metrics(Expected_Output,TR))
    prev_cost = curr_cost    
    print("Backward propagating")
    W1GRAD,W2GRAD = backpropagate(Expected_Output,TR,TD,A2,Z2,len(TD))
    
    for i in range(len(WL1)):
        for j in range(len(WL1[0])):
            WL1[i][j] = WL1[i][j] - W1GRAD[i][j]
    for i in range(len(WL2)):
        for j in range(len(WL2[0])):
            WL2[i][j] = WL2[i][j] - W2GRAD[i][j]            



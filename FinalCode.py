#References
#No of Hidden Layers and Nodes : http://goo.gl/OdBRY2
#No of Hidden Layers and Nodes : http://goo.gl/Nb4GZp


#Header Files and Imports
import random
import math
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
def initializeRandomWeights(row,col):
    weights= []
    for i in range(0,row):
        r = []
        for j in range(0, col):
            r.append(random.uniform(10, 11))
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


# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return 1/(1+pow(math.e,-1*x))

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def forwardpropagate(XL1,Y,WL1,WL2):
    XL2 = MatMul(XL1,WL1)
    ZL2 = XL2
    for i in range(len(XL2)):
        for j in range(len(XL2[0])):
            XL2[i][j] = sigmoid(XL2[i][j])
    Output = MatMul(XL2,WL2)
    ZL3 = Output
    for i in range(len(Output)):
        for j in range(len(Output[0])):
            Output[i][j] = sigmoid(Output[i][j])  
            if Output[i][j] <0 or Output[i][j] >1:
                print(Output[i][j])
    return (Output,XL2,ZL2,ZL3)


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
                JVal += (actual[i][j]*math.log(expected[i][j]) + (1-actual[i][j])*math.log(1-expected[i][j]))
    return ((JVal*1.0)/inputCount)  

#Adjust the weights by back propagating the errors in each layer        
def backpropagate(expected,actual,XL1,XL2,Z2,m):
    dL3  = expected
    for i in range(len(expected)):
        dL3[i][0] = (expected[i][0] - actual[i][0])
    print("Calculaing Delta-2")    
    dL2  = MatMul(dL3,transpose(WL2))
    for i in range(len(dL2)):
        for j in range(len(dL2[0])):
            dL2[i][j] = dL2[i][j]*dsigmoid(Z2[i][j])
    print("Calculaing Delta-Caps")        
    DL2 = MatMul(transpose(XL2),dL3)
    DL1 = MatMul(transpose(XL1),dL2)
    W1GRAD = DL1
    W2GRAD = DL2        
    for i in range(len(DL1)):
        for j in range(len(DL1[0])):
            W1GRAD[i][j] = (1.0/m)*DL1[i][j]
    for i in range(len(DL2)):
        for j in range(len(DL2[0])):
            W2GRAD[i][j] = (1.0/m)*DL2[i][j]
    return (W1GRAD,W2GRAD)        

#Input Raw Data and get Preprocessed Data
TD, TR = prepareInput('Datasets/1332/1332- active.txt','Datasets/1332/1332- inactive.txt')

#Intialize the weights for both layers
rows = len(TD[0])
cols = len(TD)/2
WL1 = initializeRandomWeights(rows,cols)
WL2 = initializeRandomWeights(cols,1)

prev_cost = 1111111111
while 1:
    print("Forward propagating")
    Expected_Output , A2 , Z2 , Z3 = forwardpropagate(TD,TR,WL1,WL2)
    print("Calculaing Cost")
    curr_cost = cost(TR,Expected_Output,len(TR))
    print(curr_cost)
    if prev_cost ==  curr_cost:
        break
    prev_cost = curr_cost    
    print("Backward propagating")
    W1GRAD,W2GRAD = backpropagate(Expected_Output,TR,TD,A2,Z2,len(TD))
    for i in range(len(WL1)):
        for j in range(len(WL1[0])):
            WL1[i][j] = WL1[i][j] - W1GRAD[i][j]
    for i in range(len(WL2)):
        for j in range(len(WL2[0])):
            WL2[i][j] = WL2[i][j] - W2GRAD[i][j]            




#References
#No of Hidden Layers and Nodes : http://goo.gl/OdBRY2
#No of Hidden Layers and Nodes : http://goo.gl/Nb4GZp


#Header Files and Imports
import random

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
            split_row = curr_row_inact.split()
            #Remove the first column. We dont need ID of the compound here
            split_row = split_row[1:]
            DataMatrix.append(split_row)
            OutputMatrix.append([0])
        if curr_row_act:
            split_row = curr_row_act.split()
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


#Input Raw Data and get Preprocessed Data
TD, TR = prepareInput('Datasets/1332/1332- active.txt','Datasets/1332/1332- inactive.txt')

#Intialize the weights for both layers
rows = len(TD[0])
cols = len(TD)/2
WL1 = initializeRandomWeights(rows,cols)
WL2 = initializeRandomWeights(rows,cols)




#Libraries to parse xls docs
from xlrd import open_workbook,cellname
import csv

#Libraries to create xlx files for further use and data preparation
import xlsxwriter

#Open data sheet
def open_sheet(data_sheet):
	book = open_workbook(data_sheet)
	#Index data sheet
	sheet = book.sheet_by_index(0)
	return sheet




# Back-Propagation Neural Networks

import math
import random
import string

#Global Variables


random.seed(0)
#Formulaes for Data Metrics

def recall(TP,FN):
    return (TP*1.0)/(TP+FN)

def pecision(TP,FP):
    return (TP*1.0)/(TP+FP)

def accuracy(TP,FP,TN,FN):
    return ((TP+TN)*1.0)/(TP+FP+TN+FN)

def F_score(p,r):
    return 2*p*r/(p+r)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
		FP  = 0
		FN = 0
		TP = 0
		TN = 0
		for p in patterns:
			res_list = self.update(p[0])
			res = res_list[0]
			print(res)
			if res >= 0.5:
				res = 1
			else:
				res = 0	
			if p[1][0] == 1 and res == 0:
				FN+=1
			elif p[1][0] == 1 and res == 1:
				TP+=1
			elif p[1][0] == 0 and res == 0:	
				TN+=1
			elif p[1][0] == 0 and res == 1:
				FP+=1
		return (FP,FN,TP,TN)
    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)

def demo():
    # Teach network XOR function

	#Read the training Data - active compounds
	pat = []
	sheet = open_sheet('Datasets/1332/1332-active.xlsx')

    #fill the pattern with data sets
	for row_index in range(1,sheet.nrows):
		descriptors = []
		for col_index in range(1,sheet.ncols):
			descriptors.append(sheet.cell(row_index,col_index).value)
		pat.append([descriptors,[1]])	

    #Read the training Data - active compounds
	sheet = open_sheet('Datasets/1332/1332-inactive.xlsx')

    #fill the pattern with data sets
	for row_index in range(1,sheet.nrows):
		descriptors = []
		for col_index in range(1,sheet.ncols):
			descriptors.append(sheet.cell(row_index,col_index).value)
		pat.append([descriptors,[0]])	


    # create a network with n input, n/2 hidden, and one output nodes
	n = NN(sheet.ncols-1, 10 , 1)
    # train it with some patterns
	n.train(pat)
    # test it

	FP,FN,TP,TN = n.test(pat)
	print("False Positives  : ",FP)
	print("True Positives  : ",TP)
	print("False Negitives  : ",FN)
	print("True Negitives  : ",TN)

if __name__ == '__main__':
	demo()
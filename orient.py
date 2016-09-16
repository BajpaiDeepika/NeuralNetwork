#                                                      Report
# Step1:K Nearest Neighhbor method:
# The approach is to first read the train-data.txt file . 
# The first word gives image name, second word gives orientation, and next 192 words give the vector details.
#We are using two read data function, one for feature vectors and actual outut as two merged list and one for reading the image name
# The task is to then read each line from test-data.txt and compare the Total vector distance (difference of 192 test vectors from respective 192 train vector )
# from each import train image and identify the K nearest distance neighbors. 
# Then use the maximum likelihood to identify the orientation of the each test image.This will be as output in
# a test file called knn_output.txt.Similarly other reports generated are nnet_output.txt and Best_output.test
#    
# Euclidean distance:
#  if p = (p1, p2,..., p192) and q = (q1, q2,..., q192) are two points
#  d(p,q)=sqrt(((q1-p1)^2)+...+((q192-p192)^2)). We are ignoring sqrt and pow for speed.
#  We will consider the absolute difference only.
#
# Classification Accuracy:
# We need to show the percentage of correctly classified image=(Total no of images-Total no of correctly identified images)/Total no of images

# Confusion matrix:
#     0 90 180 270      # row shows correct label, column shows identified label
# 0   
# 90  1                 # means 1 image had orientation as 90 and identified as 0
# 180
# 270
#For this ,we will iterate over the length of test set to check if predicted=actual,increase the respective diagonal count.
#After this, we will count the diagonal elements and sum them to calculate accuracy.

# Step 2:Neural Network Classification
# The  input layer will have 192 inputs nodes to receive the 192 vectors 
# for each image record and the weights will be assigned to these nodes randomly in the begining.I am normalizing these wights by dividing all the vectors by the highest vector to keep the values between 0-1.
# The activation function used is tanh.
# As the out is classified in one of the four orientations so the output layer will have four nodes having values as 0 or 1 .For the correct classification, it will assign the value as 1 to the highest index.
# The hidden layer will have any given number of node.so the output generated for each set will be like[1,0,0,1] -for 0, [0,1,0,0]-for 90
# [0,0,1,0]-for 270 and [0,0,0,1] for 360
#
#Issues:
# The biggest issue is the execution-speed. For neural network, if I run the test case with few data and hidden layer=1 or, the performance is bad but it takes less time. More the number of iterations, hidden layer node and length of tets and tarin data. PErformance increases but speed becomes very slow.
#Reference:
# 1. http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
# 2. http://www.bogotobogo.com/python/python_Neural_Networks_Backpropagation_for_XOR_using_one_hidden_layer.php
# 3. http://arctrix.com/nas/python/bpnn.py
# 4. Discussed with Saurabh Tripathy userId:saurtrip
import sys
import math
import operator
import random
import string   
import time
random.seed(0)
def read_dataKNN(fname):
    Set=[]
    file = open(fname, 'r');
    for line in file:
            
        y = line.split()
        Set.append((y))
#     print Set
    return Set
    
def DistanceCalc(instance1, instance2, length):
    distance = 0
    for x in range(2,length):
        distance += abs(int(instance1[x]) - int(instance2[x]))
    return distance
 
def NeighborsCalc(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)
    for l in range(len(trainingSet)):
        dist = DistanceCalc(testInstance, trainingSet[l], length)
        distances.append((trainingSet[l], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for l in range(k):
        neighbors.append(distances[l][0])
    #print neighbors[1][1]
    return neighbors
 
def ResponseCalc(neighbors):
    Orient=[0,0,0,0]
    for x in range(len(neighbors)):
        if int(neighbors[x][1])==0:
            Orient[0]=Orient[0]+1
        elif int(neighbors[x][1])==90:
            Orient[1]=Orient[1]+1
        elif int(neighbors[x][1])==180:
            Orient[2]=Orient[2]+1
        elif int(neighbors[x][1])==270:
            Orient[3]=Orient[3]+1
    #print Orient
    maxOrientIndex=Orient.index(max(Orient))
    if maxOrientIndex==0:
        return '0'
    elif maxOrientIndex==1:
        return '90'
    elif maxOrientIndex==2:
        return '180'
    elif maxOrientIndex==3:
        return '270'


def KNN_Final(knnK,train,test):
    print "K Nearest Neighbours calculation started.Please wait.."
    trainfname=train
    testfname=test
    trainingSet=[]
    testSet=[]
    trainingSet=read_dataKNN(trainfname)
    testSet=read_dataKNN(testfname)
    predicted=[]
    ImageName=[]
    k =knnK
    actual=[]
    for x in range(len(testSet)):
        ImageName.append(testSet[x][0])
        neighbors = NeighborsCalc(trainingSet, testSet[x], k)
        result = ResponseCalc(neighbors)
        predicted.append(result)
        #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][1]))
        actual.append(testSet[x][1])
    actual = map(int, actual)
    predicted = map(int, predicted)
    print "actual" ,actual
    print "predicted", predicted
    m=create_conf_matrix(actual, predicted,4)
    print ("Confusion Matrix is %s " % m)
    accurate=calc_accuracy(m)
    print ("Accuracy is %s percent " % accurate)
#*********************Printing on New Output Text File "knn_utput.txt"*********************************************************************************** 
    filename ="knn_output.txt"
    NewOutputFile = open(filename, 'w')
    for l in range(len(predicted)):
        #NewOutputFile.write("%s %d" % ImageName[l],predicted[l])
        NewOutputFile.write("%s %s\n" % (ImageName[l],predicted[l]))
    
    NewOutputFile.close()
    
def KNN_Best(knnK,train,test):
    print "Best calculation started.Please wait.."
    trainfname=train
    testfname=test
    trainingSet=[]
    testSet=[]
    trainingSet=read_dataKNN(trainfname)
    testSet=read_dataKNN(testfname)
    predicted=[]
    ImageName=[]
    k =knnK
    actual=[]
    for x in range(len(testSet)):
        ImageName.append(testSet[x][0])
        neighbors = NeighborsCalc(trainingSet, testSet[x], k)
        result = ResponseCalc(neighbors)
        predicted.append(result)
        #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][1]))
        actual.append(testSet[x][1])
    actual = map(int, actual)
    predicted = map(int, predicted)
    print "actual" ,actual
    print "predicted", predicted
    m=create_conf_matrix(actual, predicted,4)
    print ("Confusion Matrix is %s " % m)
    accurate=calc_accuracy(m)
    print ("Accuracy is %s percent " % accurate)
#*********************Printing on New Output Text File "Best_output.txt"*********************************************************************************** 
    filename ="Best_output.txt"
    NewOutputFile = open(filename, 'w')
    for l in range(len(predicted)):
        #NewOutputFile.write("%s %d" % ImageName[l],predicted[l])
        NewOutputFile.write("%s %s\n" % (ImageName[l],predicted[l]))
    
    NewOutputFile.close()

#*************************************************************************************************************************************************
#**************************************************Neural Network********************************************************************************
# Back-Propagation Neural Networks

#Refernce:http://arctrix.com/nas/python/bpnn.py




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
def read_ImageName(fname):
    Set1=[]
    ImageName=[]
    file = open(fname, 'r');
    for line in file:
        y = line.split()
        Set1.append((y))
    for j in range(len(Set1)):
        ImageName.append(Set1[j][0])
    return ImageName  
def read_data(fname):
    list1 = []
    list2=[]
    file = open(fname, 'r');
    for line in file:
        data = tuple([w.lower() for w in line.split()])
        data=list(data)
    

        
        if(data[1]=='0'):
            Op=[1,0,0,0]

        elif(data[1]=='90'):
            Op=[0,1,0,0]

        elif(data[1]=='180'):
            Op=[0,0,1,0]

        elif(data[1]=='270'):
            Op=[0,0,0,1]
        list1=[]
        for i in range(2,len(data)):
            #norm = [float(i)/max(data) for i in data]
            list1.append(float(data[i]))
        normed=[i/max(list1) for i in list1]
        list2.append([normed,Op])
        del list1

    return list2


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
        self.predicted=[]
        for p in patterns:
            #print p[0]
            if self.update(p[0]).index(max(self.update(p[0])))==0:
                #print "Predicted=0"  
                self.predicted.append(0)         
            elif self.update(p[0]).index(max(self.update(p[0])))==1:
                #print "Predicted=90"
                self.predicted.append(90)
            elif self.update(p[0]).index(max(self.update(p[0])))==2:
                #print "Predicted=180"
                self.predicted.append(180)
            elif self.update(p[0]).index(max(self.update(p[0])))==3:
                #print "Predicted=270"
                self.predicted.append(270)
          
        return self.predicted
    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=10000, N=.5, M=0.2):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
#             if i % 20 == 0:
#                 print('error %-.5f' % error)
def create_conf_matrix(expected, predicted, n_classes):
    m = [[0] * n_classes for i in range(n_classes)]
    e=expected
    p=predicted
    
    for i in range(len(e)):
        #1st Row
        if(e[i]==0 and p[i]==0):
            m[0][0]=m[0][0]+1
        elif(e[i]==0 and p[i]==90):
            m[0][1]=m[0][1]+1
        elif(e[i]==0 and p[i]==180):
            m[0][2]=m[0][2]+1
        elif(e[i]==0 and p[i]==270):
            m[0][3]=m[0][3]+1
        # 2nd Row
        if(e[i]==90 and p[i]==0):
            m[1][0]=m[1][0]+1
        elif(e[i]==90 and p[i]==90):
            m[1][1]=m[1][1]+1
        elif(e[i]==90 and p[i]==180):
            m[1][2]=m[1][2]+1
        elif(e[i]==90 and p[i]==270):
            m[1][3]=m[1][3]+1  
        #3rd row:
        if(e[i]==180 and p[i]==0):
            m[2][0]=m[2][0]+1
        elif(e[i]==180 and p[i]==90):
            m[2][1]=m[2][1]+1
        elif(e[i]==180 and p[i]==180):
            m[2][2]=m[2][2]+1
        elif(e[i]==180 and p[i]==270):
            m[2][3]=m[2][3]+1  
        #4th row:
        if(e[i]==270 and p[i]==0):
            m[3][0]=m[3][0]+1
        elif(e[i]==270 and p[i]==90):
            m[3][1]=m[3][1]+1
        elif(e[i]==270 and p[i]==180):
            m[3][2]=m[3][2]+1
        elif(e[i]==270 and p[i]==270):
            m[3][3]=m[3][3]+1    
    return m

def calc_accuracy(conf_matrix):
    acc=0
    t = sum(sum(l) for l in conf_matrix)
    if t==0:
        print "No test data-Empty Confusion matrix"
        return 190
    else:
        #print "wow"
        #print conf_matrix
        for m in range(len(conf_matrix)):
            #print conf_matrix[m][m]
          
            acc=acc+conf_matrix[m][m]
   
    #return float(acc*100)/t
    return sum(conf_matrix[i][i] for i in range(len(conf_matrix))*100) / t   
  

def demo(hn,train,test):     
    
    train_file=train
    test_file=test
    train_data = read_data(train_file)
    ImageName=read_ImageName(test_file)
    #print ImageName
    testdata=read_data(test_file)
    #print len(train_data)
    #print len(testdata) 
    print "The neural network is currently learning and takes about 1-3 hour depending on the test and train data.Please wait.."

    # create a network with two input, two hidden, and one output nodes
    
    n = NN(192,hn,4)    
    actual=[]
    p=[]
    n.train(train_data)
    
    
    #print testdata
    predicted=n.test(testdata)
      
    for i in range(len(testdata)):
        
        if testdata[i][1].index(max(testdata[i][1]))==0:
            #print "Actual=0"
            actual.append(0)
        if testdata[i][1].index(max(testdata[i][1]))==1:
            #print "Actual=90"
            actual.append(90)
        if testdata[i][1].index(max(testdata[i][1]))==2:
            #print "Actual=180"
            actual.append(180)
        if testdata[i][1].index(max(testdata[i][1]))==3:
            #print "Actual=270"
            actual.append(270)
    print "actual" ,actual
    print "predicted", predicted
    m=create_conf_matrix(actual, predicted, 4)
    print ("Confusion Matrix is %s " % m)
    accurate=calc_accuracy(m)
    print ("Accuracy is %s percent " % accurate)
    
    
#*********************Printing on New Output Text File "nn_output.txt"*********************************************************************************** 
    
    filename ="nn_output.txt"
    NewOutputFile = open(filename, 'w')
    for l in range(len(predicted)):
        #NewOutputFile.write("%s %d" % ImageName[l],predicted[l])
        NewOutputFile.write("%s %s\n" % (ImageName[l],predicted[l]))
    
    NewOutputFile.close()

def start(Algo,knnK,train,test):
    start_time = time.time()
    #print len(train),len(test)	
    knnK=knnK
    #Algo="KNN"
    if Algo=="KNN":
        KNN_Final(knnK,train,test)
    elif (Algo=="NN"):
        demo(hn,train,test)
    elif (Algo=="best"):
        KNN_Best(knnK,train,test)
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
#*************************Start Of Program***************************************8

#sys.argv[4]=3
if sys.argv[3]=="knn":
    
    Algo="KNN"
    knnK=sys.argv[4]
    knnK=int(knnK)
    train=sys.argv[1]
    test=sys.argv[2]
    start(Algo,knnK,train,test)
elif sys.argv[3]=="nnet":
    Algo="NN"
    hn=sys.argv[4]
    hn=int(hn)
    train=sys.argv[1]
    test=sys.argv[2]
    start(Algo,hn,train,test)
elif sys.argv[3]=="best":
    Algo="best"
    #hn=sys.argv[4]
    knnK=5
    train=sys.argv[1]
    test=sys.argv[2]
    start(Algo,knnK,train,test)
    
else:
    print "Please select only one algo amongts knn or nnet ot best"



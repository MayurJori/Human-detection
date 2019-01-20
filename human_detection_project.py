import os
import numpy as np
from matplotlib import pyplot
import numpy
import cv2
import math
import csv
from skimage.io import imread,imsave
import numpy as np
import sys
np.set_printoptions(threshold=np.nan)

#saving csv file for results
def savecsv(file,filename):
    with open(filename+".csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(file)

class hog_calculation:
    def __init__(self,image,path):
        self.path='./'+path+'/'+image
        self.img_name=image
        self.img=imread(self.path,dtype=float) #read image
        self.height=len(self.img)
        self.width=len(self.img[0])
        self.gray_img=np.zeros((self.height,self.width)) #initialize gray image
        self.prewitt_mat_x=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        self.prewitt_mat_y=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        self.start=int(len(self.prewitt_mat_x)/2) #self.start for removing the edge pixels
        self.height_end=self.height-self.start #self.height_end for removing the edge pixels
        self.width_end=self.width-self.start #self.height_end for removing the edge pixels
        self.gradx=np.zeros((self.height,self.width),dtype=float) 
        self.grady=np.zeros((self.height,self.width),dtype=float)
        self.grad_magnitude=np.zeros((self.height,self.width),dtype=float)
        self.grad_angle=np.zeros((self.height,self.width),dtype=float)
        self.cell_length=8
        self.cell_width=8

    def grayscale_calc(self):
        for i in range(0,self.height):
            for j in range(0,self.width):
                #grayscale image calculation from color image
                self.gray_img[i,j]=np.round(0.299*self.img[i,j,0] + 0.587*self.img[i,j,1] + 0.114*self.img[i,j,2])

    def calcgrad(self,xval,yval):
        tempgradx=0.0
        tempgrady=0.0
        #gradient calculation for each pixel
        tempgradx= (np.sum(np.multiply(self.gray_img[xval-1:xval+2,yval-1:yval+2],self.prewitt_mat_x)))/3
        tempgrady= (np.sum(np.multiply(self.gray_img[xval-1:xval+2,yval-1:yval+2],self.prewitt_mat_y)))/3
        return tempgradx,tempgrady

    #this function call calcgrad() for each pixel location
    def gradientcall(self):
        for i in range(0,self.height):
            for j in range(0,self.width):
                #if the pixels are outside the range, making gradient angle and gradient magnitude as 0
                if i<self.start or i>=self.height_end or j<self.start or j>=self.width_end:
                    self.grad_magnitude[i][j]=0
                    self.grad_angle[i][j]=0
                else:
                #calculate gradient values    
                    self.gradx[i][j],self.grady[i][j]= self.calcgrad(i,j)
                    grad_magnitude_val=(self.gradx[i][j]**2)+(self.grady[i][j]**2)
                    self.grad_magnitude[i][j]=(np.sqrt(grad_magnitude_val)/math.sqrt(2))
                    val=(np.arctan2(self.grady[i][j],self.gradx[i][j])*(180/np.pi))
                    if val<-10:
                        val+=360
                    if val>=170 and val<350:
                        val-=180
                    self.grad_angle[i][j]=val
        grad_magnitude_img=cv2.imwrite(self.img_name+'_magnitude.bmp',self.grad_magnitude)
        #initializing the bins for storing the cell values calculated later
        self.no_of_bins=9
        self.bins_length=int(len(self.grad_angle)/self.cell_length)
        self.bins_width=int(len(self.grad_angle[0])/self.cell_width)
        self.bins=np.zeros((self.bins_length,self.bins_width,self.no_of_bins)) #3D array of 20x12x9
        
        #initializing the block_vector for storing the block magnitude values 
        self.blocks_length=self.bins_length-1
        self.blocks_width=self.bins_width-1
        self.vector=np.zeros((self.blocks_length,self.blocks_width,36)) #3D array of 19x11x36

        
    #The magnitude is divided into bins according to the gradient angl value
    def calc_histogram(self,ival,jval,bins_x,bins_y):
        #for each cell in x direction
        for x in range(ival,ival+8):
            #for each cell in y direction
            for y in range(jval,jval+8):
                #getting gradient angle and gradient magnitude values for calculation
                temp_angle_val=self.grad_angle[x,y]
                temp_mag_val=self.grad_magnitude[x,y]
                #checking for the bin in which the value lies, by checking the endpoints of a bin
                #bin_val starts from 0 to 8
                if -10<=temp_angle_val<10:
                    center=0
                    bin_val=0
                    #if the angle value matches center of the bin, allocate complete magnitude value to that bin
                    if temp_angle_val==center:
                        self.bins[bins_x,bins_y,bin_val]+=temp_mag_val
                    else:
                        #calculating abolute value for diving the magnitude value in proportion
                        #high_diff is difference from right end of the bin
                        high_diff=abs(10-temp_angle_val)
                        #low_diff is difference from right end of the bin
                        low_diff=abs(-10-temp_angle_val)
                        #if the current value is less than center value of bin, divide the magnitude into current bin and previous bin
                        if temp_angle_val<center:
                            self.bins[bins_x,bins_y,bin_val]+=(high_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val-1]+=(low_diff/20)*temp_mag_val
                        #if the current value is greater than center value of bin, divide the magnitude into current bin and next bin
                        else:
                            self.bins[bins_x,bins_y,bin_val]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val+1]+=(high_diff/20)*temp_mag_val
                elif 10<=temp_angle_val<30:
                    center=20
                    bin_val=1
                    if temp_angle_val==center:
                        self.bins[bins_x,bins_y,bin_val]+=temp_mag_val
                    else:
                        high_diff=abs(30-temp_angle_val)
                        low_diff=abs(10-temp_angle_val)
                        if temp_angle_val<center:
                            self.bins[bins_x,bins_y,bin_val-1]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val]+=(high_diff/20)*temp_mag_val
                        else:
                            self.bins[bins_x,bins_y,bin_val]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val+1]+=(high_diff/20)*temp_mag_val
                elif 30<=temp_angle_val<50:
                    center=40
                    bin_val=2
                    if temp_angle_val==center:
                        self.bins[bins_x,bins_y,bin_val]+=temp_mag_val
                    else:
                        high_diff=abs(50-temp_angle_val)
                        low_diff=abs(30-temp_angle_val)
                        if temp_angle_val<center:
                            self.bins[bins_x,bins_y,bin_val-1]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val]+=(high_diff/20)*temp_mag_val
                        else:
                            self.bins[bins_x,bins_y,bin_val]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val+1]+=(high_diff/20)*temp_mag_val
                    
                elif 50<=temp_angle_val<70:
                    center=60
                    bin_val=3
                    if temp_angle_val==center:
                        self.bins[bins_x,bins_y,bin_val]+=temp_mag_val
                    else:
                        high_diff=abs(70-temp_angle_val)
                        low_diff=abs(50-temp_angle_val)
                        if temp_angle_val<center:
                            self.bins[bins_x,bins_y,bin_val-1]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val]+=(high_diff/20)*temp_mag_val
                        else:
                            self.bins[bins_x,bins_y,bin_val]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val+1]+=(high_diff/20)*temp_mag_val
                    
                elif 70<=temp_angle_val<90:
                    center=80
                    bin_val=4
                    if temp_angle_val==center:
                        self.bins[bins_x,bins_y,bin_val]+=temp_mag_val
                    else:
                        high_diff=abs(90-temp_angle_val)
                        low_diff=abs(70-temp_angle_val)
                        if temp_angle_val<center:
                            self.bins[bins_x,bins_y,bin_val-1]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val]+=(high_diff/20)*temp_mag_val
                        else:
                            self.bins[bins_x,bins_y,bin_val]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val+1]+=(high_diff/20)*temp_mag_val
                elif 90<=temp_angle_val<110:
                    center=100
                    bin_val=5
                    if temp_angle_val==center:
                        self.bins[bins_x,bins_y,bin_val]+=temp_mag_val
                    else:
                        high_diff=abs(110-temp_angle_val)
                        low_diff=abs(90-temp_angle_val)
                        if temp_angle_val<center:
                            self.bins[bins_x,bins_y,bin_val-1]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val]+=(high_diff/20)*temp_mag_val
                        else:
                            self.bins[bins_x,bins_y,bin_val]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val+1]+=(high_diff/20)*temp_mag_val
                elif 110<=temp_angle_val<130:
                    center=120
                    bin_val=6
                    if temp_angle_val==center:
                        self.bins[bins_x,bins_y,bin_val]+=temp_mag_val
                    else:
                        high_diff=abs(130-temp_angle_val)
                        low_diff=abs(110-temp_angle_val)
                        if temp_angle_val<center:
                            self.bins[bins_x,bins_y,bin_val-1]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val]+=(high_diff/20)*temp_mag_val
                        else:
                            self.bins[bins_x,bins_y,bin_val]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val+1]+=(high_diff/20)*temp_mag_val
                elif 130<=temp_angle_val<150:
                    center=140
                    bin_val=7
                    if temp_angle_val==center:
                        self.bins[bins_x,bins_y,bin_val]+=temp_mag_val
                    else:
                        high_diff=abs(150-temp_angle_val)
                        low_diff=abs(130-temp_angle_val)
                        if temp_angle_val<center:
                            self.bins[bins_x,bins_y,bin_val-1]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val]+=(high_diff/20)*temp_mag_val
                        else:
                            self.bins[bins_x,bins_y,bin_val]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val+1]+=(high_diff/20)*temp_mag_val
                elif 150<=temp_angle_val<=170:
                    center=160
                    bin_val=8
                    if temp_angle_val==center:
                        self.bins[bins_x,bins_y,bin_val]+=temp_mag_val
                    else:
                        high_diff=abs(170-temp_angle_val)
                        low_diff=abs(150-temp_angle_val)
                        if temp_angle_val<center:
                            self.bins[bins_x,bins_y,bin_val-1]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,bin_val]+=(high_diff/20)*temp_mag_val
                        else:
                            self.bins[bins_x,bins_y,bin_val]+=(low_diff/20)*temp_mag_val
                            self.bins[bins_x,bins_y,0]+=(high_diff/20)*temp_mag_val
    #This function calls calc_histogram to divide the magnitude values into respective bins
    def histogram_call(self):
        img_length=len(self.grad_angle)
        img_width=len(self.grad_angle[0])
        bins_x=0
        bins_y=0
        #for each cell in x directtion
        for i in range(0,img_length,8):
            #for each cell in y directtion
            for j in range(0,img_width,8):
                self.calc_histogram(i,j,bins_x,bins_y)
                bins_y+=1
            bins_x+=1
            bins_y=0
 
    #This function concatenates 4 cells to form a block    
    def preL2norm(self):
        count=0
        #for each block in x direction
        for i in range(0,self.blocks_length):
            #for each block in y direction
            for j in range(0,self.blocks_width):
                temp_vec=np.concatenate((self.bins[i,j],self.bins[i,j+1],self.bins[i+1,j],self.bins[i+1,j+1]), axis = None)
                self.vector[i,j]=np.copy(temp_vec)

    #This function performs normalization of each block
    def l2norm(self):
        #for each block in x direction
        for i in range(0,len(self.vector)):
        #for each block in y direction
            for j in range(0,len(self.vector[0])):
                #Calculating the normalized value for division operation
                normval=np.sqrt(np.sum(self.vector[i,j]**2))
                #divind each block values by the normalized value
                self.vector[i,j]/=normval

class Neural_Network:
    def __init__(self,size,w1,w2,b1,b2):
        self.b1=b1 #bias for hidden layer
        self.b2=b2 #bias for output layer
        self.size=size #it contains the list of 3 elements with input layer size, hidden layer size and output layer size
        self.inputLayerSize=size[0]
        self.hiddenLayerSize=size[1]
        self.outputLayerSize=size[2]
        self.w1=w1 #weights between input and hidden layer
        self.w2=w2 #weights between hidden and output layer

    #Rectified Linear Unit calculation for forward propogation in hidden layer
    def relu(self,data):
        #Values less than 0 in input data are suppressed to 0
        data[data<0]=0
        return data
    #This function calculates the derivative value of input using relu function. Used in backpropogation hidden layer 
    def reludash(self,data):
        #Values less than 0 are suppressed to zero
        data[data<0]=0
        #Values greater than zero are marked as 1
        data[data>0]=1
        return data

    #This function calculates the sigmoid value for forward propogation at output layer
    def sigmoid(self,data):
        return 1/(1+np.exp(-data))

    #This function calculates the derivative of sigmoid value for backpropogation at output layer
    def sigmoiddash(self,data):
        return (np.exp(-data)/((1+np.exp(-data))**2))

    #This is forward propogation
    def forward(self,inp):
        #calculate dot product of input and weights, add bias to the result
        self.z2=np.dot(inp.T,self.w1)+self.b1.T
        #perform activation function at hidden layer
        self.a2=self.relu(self.z2)
        #pass result of activation to output layer by multiplying by weights and adding bias
        self.z3=np.dot(self.a2,self.w2)+self.b2
        #perform activation at output layer
        self.yHat=self.sigmoid(self.z3)
        return self.yHat

    #This is backward propogation
    def backward(self,inp,y,original_inp):
            #calculate error at output layer
            self.delta3 = np.multiply(self.yHat-y,self.sigmoiddash(self.z3))
            #calculate error at hiddenlayer
            self.dJdW2 = np.dot(self.a2.T,self.delta3)
            #calculate gradient value at output layer
            self.delta2 = np.multiply(self.delta3,self.w2.T)*self.reludash(self.z2)
            #calculate gradient value at hidden layer
            self.dJdW1 = np.dot(original_inp,self.delta2)
    
    #This function calls forward and backward propogation values and changes the weights and biases accordingly
    def costFunction(self,ip,y,lr):
            self.yHat=self.forward(ip)
            self.backward(self.yHat,y,ip)
            self.w1=self.w1+(-lr*self.dJdW1)
            self.w2=self.w2+(-lr*self.dJdW2)
            self.b1=self.b1+(-lr*self.delta2.T)
            self.b2=self.b2+(-lr*self.delta3)
            return self.w1,self.w2,self.b1,self.b2,self.yHat

    #Testing data is passed here and output value is calculated        
    def testing_function(self,inp_data):
        self.yHat=self.forward(inp_data)
        return self.yHat


#training code calculates the weights and biases for testing data
def training_code():
    global iterations
    #my_data (20x7524) contains the HoG data of 20 images out of which first 10 are positive and last 10 are negative
    my_data = np.genfromtxt('train_x.csv', delimiter=',')

    #weights and biases are randomly initialized
    w1 = np.random.randn(inputLayerSize,hiddenLayerSize)*0.01
    w2 = np.random.randn(hiddenLayerSize,outputLayerSize)*0.01
    b1=np.random.randn(hiddenLayerSize,outputLayerSize)
    b2=np.random.randn(outputLayerSize,outputLayerSize)

    #This array contained the target values for training data
    train_y=np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])

    #variables for checking the difference between RMS value of error between successive epochs
    error_val=0.0
    old_error_val=0.1

    #while the difference between successive RMS value of error is greater than some threshold value, we perform training
    while abs(old_error_val-error_val)>0.0000001:
        global iterations
        iterations+=1
        old_error_val=error_val
        error_val=0
        y_val=0
        #for each image HoG in training data
        for row in my_data:
            #copy that image into inputlayer variable
            inputlayer=np.copy(row)
            #reshaping it for our operations
            inputlayer=np.reshape(inputlayer,(inputLayerSize,outputLayerSize))
            #creating instance of Neural_Network for that image for calculating and updating weights
            nnobject=Neural_Network([inputLayerSize,hiddenLayerSize,outputLayerSize],w1,w2,b1,b2)
            #Updated weights are returned when done with forward and backward propogation on one image
            w1,w2,b1,b2,yHat=nnobject.costFunction(inputlayer,train_y[y_val],float(sys.argv[1]))
            #adding error value to error_val
            error_val+=(0.5*((train_y[y_val]-yHat)**2))
            y_val+=1
        #dividing error val by number of input training image count
        error_val=error_val/20
    #Saving the wight and bias values for testing data
    savecsv(w1,"final_w1")
    savecsv(w2,"final_w2")
    savecsv(b1,"final_b1")
    savecsv(b2,"final_b2")
    print ("iterations",iterations)
#this function tests the testing images for result calculation
def testing_code():
    #getting weight and bia values for testing process
    testw1 = np.genfromtxt('final_w1.csv', delimiter=',')
    testw2 = np.genfromtxt('final_w2.csv', delimiter=',')
    testb1=np.genfromtxt('final_b1.csv', delimiter=',')
    testb2=np.genfromtxt('final_b2.csv', delimiter=',')

    #reshaping values for further calculation
    testw1=np.reshape(testw1,(inputLayerSize,hiddenLayerSize))
    testw2=np.reshape(testw2,(hiddenLayerSize,outputLayerSize))
    testb1=np.reshape(testb1,(hiddenLayerSize,outputLayerSize))
    testb2=np.reshape(testb2,(outputLayerSize,outputLayerSize))
    
    #Array for storing the prediction result
    predict_y=np.zeros((10,1))

    #csv file (10x7524) containing the test images out of which first 5 are positive and last 5 are negative 
    test_x=np.genfromtxt('test_x.csv',delimiter=",")
    z=0
    #for each image HoG in testing data
    for row in test_x:
        #Create instance of Neural_Network and calculate the prediction for that image
        obj=Neural_Network([inputLayerSize,hiddenLayerSize,outputLayerSize],testw1,testw2,testb1,testb2)
        row=np.reshape(row,(inputLayerSize,outputLayerSize))

        #this function forward propogates and calculates the result
        predict_y[z]=obj.testing_function(row)
        z+=1

    #count array for storing the result
    count=[]
    #If the predicted value is greter than 0.5 for positive images then, human is present in the images
    for i in range(0,5):
        print (predict_y[i])
        if predict_y[i]>=0.5:
            count.append('1')
        else:
            count.append('0')
    #If the predicted value is less than 0.5 for positive images then, human is not present in the images
    for i in range(5,10):
        print (predict_y[i])
        if predict_y[i]<0.5:
            count.append('0')
        else:
            count.append('1')
    print (count)
#Generating the csv for training images
def csv_for_training():
    #get positive and neagative images from Train_Positive and Train_Negative folders present
    positive_train_images=os.listdir('./Train_Positive')
    negative_train_images=os.listdir('./Train_Negative')
    
    #Array for storing the training HoG values
    train_x=np.zeros((20,7524))

    j=0
    #for each positive image
    for i in range(0,len(positive_train_images)):
        # normalized HoG is calculated for an image
        obj=hog_calculation(str(positive_train_images[i]),'Train_Positive')
        obj.grayscale_calc()
        obj.gradientcall()
        obj.histogram_call()
        obj.preL2norm()
        obj.l2norm()

        #storing the normalized HoG into train_x numpy array which is later stored in csv format
        nninputlayer=len(obj.vector)*len(obj.vector[0])*len(obj.vector[0][0])
        inputlayer=np.reshape(obj.vector,(nninputlayer,1))
        train_x[j]=np.copy(inputlayer.T)
        j+=1

    #for each negative image
    for i in range(0,len(negative_train_images)):
        # normalized HoG is calculated for an image
        obj=hog_calculation(str(negative_train_images[i]),'Train_Negative')
        obj.grayscale_calc()
        obj.gradientcall()
        obj.histogram_call()
        obj.preL2norm()
        obj.l2norm()
        #storing the normalized HoG into train_x numpy array which is later stored in csv format
        nninputlayer=len(obj.vector)*len(obj.vector[0])*len(obj.vector[0][0])
        inputlayer=np.reshape(obj.vector,(nninputlayer,1))
        train_x[j]=np.copy(inputlayer.T)
        j+=1
    train_x=np.nan_to_num(train_x)

    #saving the trainx numpy array as csv file 
    savecsv(train_x,'train_x')

#Generate csv for testing images
def csv_for_testing():
    #get positive and neagative images from Test_Positive and Test_Negative folders present
    positive_test_images=os.listdir('./Test_Positive')
    negative_test_images=os.listdir('./Test_Negative')

    #Array for storing the tesing HoG values
    test_x=np.zeros((10,7524))

    #array for prediction score
    predict_y=np.zeros((10,1))

    j=0
    #for each positive image
    for i in range(0,len(positive_test_images)):
        # normalized HoG is calculated for an image
        # print (positive_test_images[i])
        obj=hog_calculation(str(positive_test_images[i]),'Test_Positive')
        obj.grayscale_calc()
        obj.gradientcall()
        obj.histogram_call()
        obj.preL2norm()
        obj.l2norm()
        #storing the normalized HoG into test_x numpy array which is later stored in csv format
        nninputlayer=len(obj.vector)*len(obj.vector[0])*len(obj.vector[0][0])
        inputlayer=np.reshape(obj.vector,(nninputlayer,1))
        test_x[j]=np.copy(inputlayer.T)
        j+=1

    #for each negative image
    for i in range(0,len(negative_test_images)):
        # normalized HoG is calculated for an image
        # print (negative_test_images[i])
        obj=hog_calculation(str(negative_test_images[i]),'Test_Negative')
        obj.grayscale_calc()
        obj.gradientcall()
        obj.histogram_call()
        obj.preL2norm()
        obj.l2norm()
        #storing the normalized HoG into test_x numpy array which is later stored in csv format
        nninputlayer=len(obj.vector)*len(obj.vector[0])*len(obj.vector[0][0])
        inputlayer=np.reshape(obj.vector,(nninputlayer,1))
        test_x[j]=np.copy(inputlayer.T)
        j+=1

    #converting nan values to zero to avoid error in calculation
    test_x=np.nan_to_num(test_x)
    # saving the testing images HoG descriptor
    savecsv(test_x,'test_x')



inputLayerSize=7524        
hiddenLayerSize=int(sys.argv[2])
outputLayerSize=1
#Iteration variable for keeping count of number of epochs
iterations=0

#these functions create csv fof HoG descriptor for training images and testing images
csv_for_training()
csv_for_testing()

#trainging and testing functions are executed here
training_code()
testing_code()


# obj=hog_calculation('crop001045b.bmp','Test_Positive')
# obj.grayscale_calc()
# obj.gradientcall()
# obj.histogram_call()
# obj.preL2norm()
# obj.l2norm()

os.system('espeak done')

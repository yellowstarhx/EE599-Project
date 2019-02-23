
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np
import os  
from skimage import transform,data
from skimage import color
train_data = np.zeros([160,784])
#print(len(result))
counter=0
for i in range (1,41):
    print("i: "+str(i))
    for j in range (1,5):
        # Read image 
        print(str(j))
        path='/Users/yueyue/Desktop/usc/M2S2/EE599/project/new_train/'+str('{:03}'.format(i))+'/'+str(j)+'.png'
        if(os.path.isfile(path)==0):
            path='/Users/yueyue/Desktop/usc/M2S2/EE599/project/new_train/'+str('{:03}'.format(i))+'/'+str(j)+'.jpg'
        img = mpimg.imread(path)
        img = color.rgb2gray(transform.resize(img, (28, 28))) #resize
        # Output Images (uncomment below two lines to check if image can be printed sucessfully)
        plt.figure(i*4+j) 
        imgplot = plt.imshow(img, cmap='gray')
        plt.show()
        img=img.flatten()
        train_data[(i-1)*4+j-1]=img
        counter+=1
print(counter)

train_label = np.zeros((160, 1))
for i in range(0,40):
    for j in range(1,5):
        train_label[i*4+j-1]=i
#print(label)

train_dataset=np.concatenate((train_label, train_data), axis=1)
np.savetxt("/Users/yueyue/Desktop/usc/M2S2/EE599/project/new_train/train.csv",train_dataset, delimiter=',')

test_data = np.zeros([40,784])
#print(len(result))
counter=0
for i in range (1,41):
    print("i: "+str(i))
    for j in range (5,6):
        # Read image 
        print(str(j))
        path='/Users/yueyue/Desktop/usc/M2S2/EE599/project/new_train/'+str('{:03}'.format(i))+'/'+str(j)+'.png'
        if(os.path.isfile(path)==0):
            path='/Users/yueyue/Desktop/usc/M2S2/EE599/project/new_train/'+str('{:03}'.format(i))+'/'+str(j)+'.jpg'
        img = mpimg.imread(path)
        img = color.rgb2gray(transform.resize(img, (28, 28))) #resize
        # Output Images (uncomment below two lines to check if image can be printed sucessfully)
        plt.figure(i+j) 
        imgplot = plt.imshow(img, cmap='gray')
        plt.show()
        img=img.flatten()
        test_data[(i-1)*1+j-5]=img
        counter+=1
print(counter)

test_label = np.zeros((40, 1))
for i in range(0,40):
    for j in range(1,2):
        test_label[i+j-1]=i

test_dataset=np.concatenate((test_label, test_data), axis=1)
np.savetxt("/Users/yueyue/Desktop/usc/M2S2/EE599/project/new_train/test.csv",test_dataset, delimiter=',')

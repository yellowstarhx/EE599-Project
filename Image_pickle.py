import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import numpy as np
import os  
import pickle
from skimage import transform,data
from skimage import color
train_data = np.zeros([320,784])
#print(len(result))
counter=0
for i in range (1,41):
    print("i: "+str(i))
    for j in range (1,9):
        # Read image 
        print(str(j))
        path='/Users/yueyue/Desktop/usc/M2S2/EE599/project/enhanced_train/'+str('{:03}'.format(i))+'/'+str(j)+'.png'
        if(os.path.isfile(path)==0):
            path='/Users/yueyue/Desktop/usc/M2S2/EE599/project/enhanced_train/'+str('{:03}'.format(i))+'/'+str(j)+'.jpg'
        img = mpimg.imread(path)
        img = color.rgb2gray(transform.resize(img, (28, 28))) #resize
        # Output Images (uncomment below two lines to check if image can be printed sucessfully)
        plt.figure(i*8+j) 
        imgplot = plt.imshow(img, cmap='gray')
        plt.show()
        img=img.flatten()
        train_data[(i-1)*8+j-1]=img
        counter+=1
print(counter)


train_label = np.zeros((320, 1))
for i in range(0,40):
    for j in range(1,9):
        train_label[i*8+j-1]=i+1
#print(label)



train_dataset=np.concatenate((train_label, train_data), axis=1)
fw = open('/Users/yueyue/Desktop/usc/M2S2/EE599/project/enhanced_train/LeNet5_train.txt','wb')
pickle.dump(train_dataset, fw)
# Pickle dictionary using protocol 0.
fw.close()
#np.savetxt("/Users/yueyue/Desktop/usc/M2S2/EE599/project/enhanced_train/LeNet5_train.csv",train_dataset, delimiter=',')

test_data = np.zeros([80,784])
#print(len(result))
counter=0
for i in range (1,41):
    print("i: "+str(i))
    for j in range (9,11):
        # Read image 
        print(str(j))
        path='/Users/yueyue/Desktop/usc/M2S2/EE599/project/enhanced_train/'+str('{:03}'.format(i))+'/'+str(j)+'.png'
        if(os.path.isfile(path)==0):
            path='/Users/yueyue/Desktop/usc/M2S2/EE599/project/enhanced_train/'+str('{:03}'.format(i))+'/'+str(j)+'.jpg'
        img = mpimg.imread(path)
        img = color.rgb2gray(transform.resize(img, (28, 28))) #resize
        # Output Images (uncomment below two lines to check if image can be printed sucessfully)
        plt.figure(i+j) 
        imgplot = plt.imshow(img, cmap='gray')
        plt.show()
        img=img.flatten()
        test_data[(i-1)*2+j-9]=img
        counter+=1
print(counter)

test_label = np.zeros((80, 1))
for i in range(0,40):
    for j in range(1,3):
        test_label[i*2+j-1]=i+1
]
test_dataset=np.concatenate((test_label, test_data), axis=1)
fw = open('/Users/yueyue/Desktop/usc/M2S2/EE599/project/enhanced_train/LeNet5_test.txt','wb')
# Pickle the list using the highest protocol available.
# Pickle dictionary using protocol 0.
pickle.dump(test_dataset, fw)
fw.close()
#np.savetxt("/Users/yueyue/Desktop/usc/M2S2/EE599/project/enhanced_train/LeNet5_test.csv",test_dataset, delimiter=',')



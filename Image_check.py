import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import numpy as np
import os  
counter=0
for i in range (1,34): # for Tianjiao, use this line
# for i in range (34,67): # for Xin, use this line
# for i in range (67,101): # for Jiayue, use this line
    #print("i:"+str(i)) #uncomment this line to check which image is not readable, if so, change a new image
    for j in range (1,6):
        # Read image 
        #print(j) #uncomment this line to check which image is not readable, if so, change a new image
        path='/Users/yueyue/Desktop/usc/M2S2/EE599/project/train/'+str('{:03}'.format(i))+'/'+str(j)+'.png'
        if(os.path.isfile(path)==0):
            path='/Users/yueyue/Desktop/usc/M2S2/EE599/project/train/'+str('{:03}'.format(i))+'/'+str(j)+'.jpg'
        img = mpimg.imread(path) 
        # Output Images (uncomment below two lines to check if image can be printed sucessfully)
        #plt.figure(i*5+j)
        #imgplot = plt.imshow(img)
        counter+=1
print(counter) # please check final print counter==33*5


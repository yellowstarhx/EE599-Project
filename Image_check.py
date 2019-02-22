import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import transform,data
from skimage import color
np_fashion_train = np.zeros([6,784])
#print(len(result))
for i in range (39,41):
    for j in range (1,4):
        # Read image
        path='/Users/yueyue/Desktop/usc/M2S2/EE599/project/new_train/'+str('{:03}'.format(i))+'/'+str(j)+'.png'
        if(os.path.isfile(path)==0):
            path='/Users/yueyue/Desktop/usc/M2S2/EE599/project/train/'+str('{:03}'.format(i))+'/'+str(j)+'.jpg'
        img = mpimg.imread(path)
        img = color.rgb2gray(transform.resize(img, (28, 28))) #resize
        # Output Images (uncomment below two lines to check if image can be printed sucessfully)
        plt.figure(i*3+j)
        imgplot = plt.imshow(img, cmap='gray')
        plt.show()
        img=img.flatten()
    np_fashion_train[(i-39)*3+j-1]=img
#np.savetxt("/Users/yueyue/Desktop/usc/M2S2/EE599/project/new_train/new.csv", result, delimiter=',')

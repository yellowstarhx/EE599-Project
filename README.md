## EE599-Project: Pokemon Recognition
Jiayue Yu & Tianjiao Feng & Xin Huang  
---  
### Updates

#### Update Apr.3 20 p.m.
Xin  
Upload KNN.cpp, KNN.h, KNNtest.cpp, KNN OOP implementation complished.  
***Compile***: `g++ KNNtest.cpp KNN.cpp data.cpp -o model`  
***Run***:     `./model k train.csv test.csv`  
(By default, **k = 5**, train.csv = LeNet5_train.csv, test.csv = LeNet5_test.csv)  
***Result***:  
  **LeNet5_train.csv** and **LeNet5_test.csv** contain *320* and *80* images respectively. Each image is a gray-scale image consisted of 28*28 pixels and each pixel is converted to 0-1.  
  `./model 1` ----> accuracy = 17.5%  
  `./model`   ----> accuracy = 15%  
  **28RGB_train.csv** and **28RGB_test.csv** contain *160* and *40* images respectively. Each image is a RGB image consisted of 28*28*3 pixels and each pixel is converted to 0-1.  
  `./model 1 28RGB_train.csv 28RGB_test.csv`  ----> accuracy = 20%  
  `./model 5 28RGB_train.csv 28RGB_test.csv`  ----> accuracy = 7.5%  


#### Update Apr.1 20 p.m.
Xin  
Upload data.cpp, data.h, datatest.cpp, which serve as C++ data structure in KNN OOP implementation.  

#### Update Mar.16 12 p.m.
Xin  
Create EnhanceDataSet.ipynb, expand images in each class to 30 by ratating & transposing

#### Update Feb.23 4 p.m.
Jiayue  
Create LeNet5_v2.py and please ignore LeNet5_v1.py

#### Update Feb.23 3 p.m.
Xin  
Update datasetMaker.ipynb, which convert the images to .ftrecords format  

#### Update Feb.22 6 p.m.
Jiayue  
Create ImagetoCSV.py and LeNet5_v1.py

#### Update Feb.22 1 p.m.
Jiayue  
Update Image_check.py with rgb-to-gray, resize and save-to-csv

#### Update Feb.16 7 p.m.
Jiayue  
Create Pokemon.py

#### Update Feb.16 1 p.m.
Jiayue  
Create Image_check.py

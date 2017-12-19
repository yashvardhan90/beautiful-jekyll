
## Principal Component Analysis and Support Vector Machine  

All the images in the dataset are in .gif format which is a sequence of frames. We need to extract the frames and convert them into a format that is suitable for matrix operations. This could be done using the Image class() from the PIL package in Python.  

First we import the dependencies.

[//]: # (Image References)

[image1]: http://github.com/yashvardhan90/yashvardhan90.github.io/raw/master/img/PCA/PCA_5_1.png
[image2]: http://github.com/yashvardhan90/yashvardhan90.github.io/raw/master/img/PCA/PCA_10_1.png
[image3]: http://github.com/yashvardhan90/yashvardhan90.github.io/raw/master/img/PCA/PCA_11_1.png
[image4]: http://github.com/yashvardhan90/yashvardhan90.github.io/raw/master/img/PCA/PCA_14_0.png
[image5]: http://github.com/yashvardhan90/yashvardhan90.github.io/raw/master/img/PCA/PCA_17_0.png
[image6]: http://github.com/yashvardhan90/yashvardhan90.github.io/raw/master/img/PCA/PCA_18_0.png


```python
import os
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
from PIL import Image
from sklearn import preprocessing
import matplotlib.pyplot as plt
%matplotlib inline
```

Then, we read the images from the dataset one-by-one and convert them to a Numpy 2-D matrix. These matrices are stored in an array `faces_data`. At the same time we also retreive the class that the image belongs to, in this case the Class ID of the person and store it in an array `class_label`.


```python
data = 'yalefaces/'
faces_data = []
class_label = []
for filename in os.listdir(data):
    if filename.endswith(".txt") or filename.endswith(".DS_Store") :
        continue
    else:
        img_array = np.array(Image.open(data+filename)).astype(float)
        img_array = cv2.resize(img_array, (80,61))
        faces_data.append(img_array)
        label = filename[7:9]
        class_label.append(label)
faces_data = np.array(faces_data)
class_label = np.array(class_label)
```


```python
plt.imshow(faces_data[0], cmap='gray')
```

    <matplotlib.image.AxesImage at 0x1167e75f8>

![alt text][image1]

```python
X = []
for img in faces_data:
    X.append(np.reshape(img, (1, img.shape[0]*img.shape[1])))
X = np.array(X)
X = np.reshape(X, (X.shape[0], X.shape[2]))
mu = np.mean(X, axis=0)
X = X - mu
```


```python
cov_matrix = np.cov(X.T)
eigval,eigvect = np.linalg.eigh(cov_matrix)
#Sorting the eigen values and eigen vectors
temp = np.argsort(eigval)
temp = temp[::-1]
eigenvec = eigvect[:,temp]  ##this ones are sorted
eigenval = eigval[temp]  ##this ones are sorted
```


```python
eigenval_ten = [eigenval[i] for i in range(10)]
eigenval_ten = np.asarray(eigenval_ten)
```


```python
plt.figure()
plt.plot(eigenval)
plt.xlabel('Principal Component Number')
plt.ylabel('Energy')
plt.title('Pricipal Component Number Vs Energy')
```


    Text(0.5,1,'Pricipal Component Number Vs Energy')




![alt text][image2]



```python
plt.figure()
plt.plot(eigenval_ten)
plt.xlabel('Principal Component Number')
plt.ylabel('Energy')
plt.title('Pricipal Component Number Vs Energy')
```


    Text(0.5,1,'Pricipal Component Number Vs Energy')


![alt text][image3]



```python
t = np.sum(eigenval)
p = 0.0
for i in range(len(eigenval)):
    p += eigenval[i]
    r = p/t
    if r > 0.5:
        break
print("To capture 50% of the energy we need the first {} principal components.".format(i+1))
```

    To capture 50% of the energy we need the first 3 principal components.



```python
plt.figure(figsize=(10,20))
eigenvec_transpose = eigenvec.T
for i in range(10):
    plt.subplot(5,2,i+1)
    plt.imshow(np.reshape(eigenvec_transpose[i],(61,80)), cmap='gray')
    plt.title('Eigenface %s'%(i+1), fontsize=14)
    plt.axis('off')
plt.tight_layout()
```


![alt text][image4]



```python
# Reconstruction of the faces
nComp = [1, 10, 20, 30, 40, 50, 100]

def plot_face(X, nComp, faceNo):
    plt.figure(figsize=(10,16))
    for i in range(len(nComp)):
        pca_space = np.dot(X,eigenvec[:,:nComp[i]])
        recon_img = np.dot(pca_space,eigenvec_transpose[:nComp[i]])
        recon_img += mu
        plt.subplot(4,2,i+1)
        plt.imshow(255*np.reshape(recon_img[faceNo], (61,80)), cmap='gray')
        plt.title('%s Components'%(nComp[i]), fontsize=14)
        plt.axis('off')
    plt.tight_layout()

```


```python
plot_face(X, nComp, 10)
```


![alt text][image5]



```python
plot_face(X, nComp, 10)
```


![alt text][image6]


It seems that we need at least 100 components to achieve a visually good result.


```python
lb = preprocessing.LabelEncoder()
y = lb.fit_transform(class_label)
```


```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

accuracy_scores = {}

pca_comp = [1,10,20,21,30,40,50,60,70,80,90,100,165]

for c in pca_comp:
    pca_space_svm = np.dot(X,eigenvec[:,:c])
    X_train, X_test, Y_train, Y_test = train_test_split(pca_space_svm,y,test_size=0.25,random_state=0)
    clf = SVC(kernel='linear')
    scores = cross_val_score(clf,X_train,Y_train,cv=5)
    accuracy_scores[c]= scores.mean()
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy for", c, "PCA componenets:",accuracy_score(Y_test, y_pred))
print("\n")
print("Cross validation scores for different PCA components:")
print(accuracy_scores)
```

    Accuracy for 1 PCA componenets: 0.214285714286
    Accuracy for 10 PCA componenets: 0.738095238095
    Accuracy for 20 PCA componenets: 0.833333333333
    Accuracy for 21 PCA componenets: 0.833333333333
    Accuracy for 30 PCA componenets: 0.833333333333
    Accuracy for 40 PCA componenets: 0.833333333333
    Accuracy for 50 PCA componenets: 0.833333333333
    Accuracy for 60 PCA componenets: 0.833333333333
    Accuracy for 70 PCA componenets: 0.833333333333
    Accuracy for 80 PCA componenets: 0.833333333333
    Accuracy for 90 PCA componenets: 0.833333333333
    Accuracy for 100 PCA componenets: 0.833333333333
    Accuracy for 165 PCA componenets: 0.833333333333
    
    
    Cross validation scores for different PCA components:
    {1: 0.26474945533769068, 100: 0.83601307189542484, 165: 0.83601307189542484, 70: 0.83601307189542484, 40: 0.84934640522875815, 10: 0.76174291938997829, 80: 0.83601307189542484, 50: 0.84267973856209155, 20: 0.82091503267973853, 21: 0.80915032679738563, 90: 0.83601307189542484, 60: 0.84267973856209155, 30: 0.84267973856209155}


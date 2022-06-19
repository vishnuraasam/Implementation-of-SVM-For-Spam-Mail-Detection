# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the necessary packages using import statement. 
2. Read the given csv file and print the number of contents to be displayed. 
3. Split the dataset using train_test_split. 
4. Calculate Y_Pred and accuracy. 
5. Display the result.  
## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: Rasam Vishnu 
RegisterNumber: 212220040131


import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
## Data.head
![8 1](https://user-images.githubusercontent.com/103240414/174470743-44414df7-397f-4027-9295-612bd37e05d7.png)
## Data.info
![8 2](https://user-images.githubusercontent.com/103240414/174470761-34361408-b3b2-4be9-9bab-4063c4cc55be.png)
## Data.isnull().sum()
![8 3](https://user-images.githubusercontent.com/103240414/174470764-71aa6a15-6a6e-4115-ae27-12bf8b739361.png)
## y_pred
![8 4](https://user-images.githubusercontent.com/103240414/174470768-407b2625-8e1e-4c68-888d-d23251c65af8.png)
## accuracy
![8 6](https://user-images.githubusercontent.com/103240414/174470776-738823bf-14fa-49ec-9988-20c652860b53.png)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

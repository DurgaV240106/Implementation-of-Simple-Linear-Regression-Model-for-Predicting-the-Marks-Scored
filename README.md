# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries for data handling, visualization, and model building.
2. Load the dataset and inspect the first and last few records to understand the data structure.
3. Prepare the data by separating the independent variable (hours studied) and the dependent
 variable (marks scored).
4. Split the dataset into training and testing sets to evaluate the model's performance.
5. Initialize and train a linear regression model using the training data.
6. Predict the marks for the test set using the trained model.
7. Evaluate the model by comparing the predicted marks with the actual marks from the test set.
8. Visualize the results for both the training and test sets by plotting the actual data points and
 the regression line 


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DURGA V
RegisterNumber: 212223230052 
*/

import numpy as np
import pandas as pd
from sklearn.metrics import  mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
## Output:
![Screenshot 2024-08-29 213840](https://github.com/user-attachments/assets/8cf32d5d-b42a-4eeb-a1e2-1a211b980705)

```
dataset.info()
```
## Output:
![Screenshot 2024-08-29 214208](https://github.com/user-attachments/assets/3a936739-7465-4fcd-adb0-6661c7dee810)

```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
## Output:
![Screenshot 2024-08-29 214344](https://github.com/user-attachments/assets/58354539-80a3-4c14-86d2-aff683a0f081)
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train.shape)
print(X_test.shape)
```
## Output:
![Screenshot 2024-08-29 214528](https://github.com/user-attachments/assets/09c39dbe-d0de-4c8d-94f2-061c6dd54c7e)
```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
```
## Output:
![Screenshot 2024-08-29 215254](https://github.com/user-attachments/assets/46c35dd1-e533-4e40-a5ba-e513f79a797a)
```
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
## Output:
![Screenshot 2024-08-29 214636](https://github.com/user-attachments/assets/5a17f6fe-ff24-42ef-96fd-c4c9811531a3)

```
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,reg.predict(X_train),color="green")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:
![Screenshot 2024-08-29 215728](https://github.com/user-attachments/assets/f8abe9d8-74f9-4bec-86a0-7468a142f58c)
```
plt.scatter(X_test, Y_test,color="blue")
plt.plot(X_test, reg.predict(X_test), color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:
![Screenshot 2024-08-29 215808](https://github.com/user-attachments/assets/03f5fb9c-894b-4ae5-932c-9454e616c367)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


diabetes_dataset = pd.read_csv("C:/Users/abhi0/Desktop/majorproject/backend/Diabetes/diabetes.csv")

# now abhi you can explain who is considered as diabetics and non diabetics
diabetes_dataset.groupby('Outcome').mean()

#separating the data and variables by
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()

scaler.fit(X)

standarized_data = scaler.transform(X)

X = standarized_data
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape,X_train.shape,X_test.shape)
#768 original out of those 648 is training and remaining for testing


#linear model support vector machine
classifier = svm.SVC(kernel='linear')

#training the support vector machine (svm)
classifier.fit(X_train,Y_train)

#accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

#printing the accuracy
print('Accuracy score of the training data : ', training_data_accuracy)
#out of 100 our model is predicting 78 times correct times(Training Data) abhi this can be done more by giving it more datasets and using it train our model more


#Now on the real time data or unknown data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

#printing the accuracy score of the real test data which is not used for the training
print('Accuracy score of the test data : ', test_data_accuracy)
#which turned out to be 77 which is pretty good and which is not overfitting

input_data = (4,110,92,0,0,37.6,0.191,30)

#changing to numpy array cause its more efficient we numpy as np
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#model excepts complete file but now we are telling that we are only giving a one set of values
#to be more clear not giving 768 files instead we are gonna give 1


#standardzing the input data here
std_data = scaler.transform(input_data_reshaped)
#print(std_data)

prediction = classifier.predict(std_data)
#print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
  
  
  
import pickle
filename = "trained_model.sav"
pickle.dump(classifier,open(filename,'wb'))
#wb writebinary (writing the classifier)

#loading the saved model here
loaded_model = pickle.load(open('C:/Users/abhi0/Desktop/majorproject/backend/Diabetes/trainedmodel.sav','rb'))
#rb readbinary

input_data = (4,110,92,0,0,37.6,0.191,30)

#changing to numpy array cause its more efficient we numpy as np
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#model excepts complete file but now we are telling that we are only giving a one set of values
#to be more clear not giving 768 files instead we are gonna give 1


#standardzing the input data here
std_data = scaler.transform(input_data_reshaped)
#print(std_data)

prediction = loaded_model.predict(std_data)
#print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
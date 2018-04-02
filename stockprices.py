import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
data = pd.read_csv('NSE-BANKBARODA.csv')
data = data.drop(['Date'],1)
data = data.drop([4929])
#print(data)
n=data.shape[0]
p=data.shape[1]
data=data.values

train_start = 0
train_end = int(np.floor(0.8 * n))
test_start = train_end+1
test_end = n
#print(train_start,train_end,test_start,test_end)

training_data = data[np.arange(train_start,train_end),:]
#print(training_data)
testing_data = data[np.arange(test_start,test_end), :]
#print(testing_data)

rf=DecisionTreeRegressor()
#print(training_data[:,[0,1,2,3]])
#print(training_data[:,[4]])

features = training_data[:,[0,1,2,3]]
labels = training_data[:,[4]]

rf.fit(features,labels)

expected = testing_data[:,[4]]
predicted = rf.predict(testing_data[:,[0,1,2,3]])
print("the expected",expected)


predicted = predicted.reshape((len(predicted),1))
print("the predicted",predicted)

print(accuracy_score(expected,predicted))
plt.plot(expected,'r')
plt.plot(predicted,'b')
plt.show()

target =[168.75,171.20,166.95,167.90]

target = np.asarray(target)

target = target.reshape(1,-1)
value = rf.predict(target)

print(value)



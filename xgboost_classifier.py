import util
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_accepted = util.read_textfile('acceptscores.txt')
X_rejection =  util.read_textfile('rejectscores.txt')

Y_accepted = np.zeros((len(X_accepted),))
Y_rejection = np.ones((len(X_rejection),))

X = np.vstack((X_accepted,X_rejection))
Y = np.hstack((Y_accepted,Y_rejection))

X_new = np.zeros((len(X),11+6))
count = 0
addon = 6

for i in X:
    X_new[count,0:6] = i
    X_new[count,addon+0] = i[i.nonzero()].mean()
    X_new[count,addon+1] = i[i.nonzero()].std()
    temp = np.ptp(i[i.nonzero()])
    if(temp == 0):
        X_new[count,addon+2] = np.max(X_new[:,addon+2])
    else:
        X_new[count,addon+2] = X_new[count,addon+0]*(1.0/(temp+0.00001))
    a = i
    X_new[count,addon+3] = np.sum(np.logical_and(a>0.1, a<=1))
    X_new[count,addon+4] = np.sum(np.logical_and(a>1, a<=2))
    X_new[count,addon+5] = np.sum(np.logical_and(a>2, a<=3))
    X_new[count,addon+6] = np.sum(np.logical_and(a>3, a<=4))
    X_new[count,addon+7] = np.sum(np.logical_and(a>4, a<=5))
    X_new[count,addon+8] = np.sum(np.logical_and(a>5, a<=6))
    X_new[count,addon+9] = np.max(i[i.nonzero()])
    X_new[count,addon+10] = np.min(i[i.nonzero()])
    count +=1
accuracy = np.zeros((1,2));
for i in np.arange(1):
	seed = np.random.randint(200, size=(1,))
	#print(seed)
	test_size = 0.2
	X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=test_size, random_state=seed[1])
	model = XGBClassifier()
	model.fit(X_train[:,6:-1], y_train)
	y_pred = model.predict(X_test[:,6:-1])
	predictions = [round(value) for value in y_pred]

	accuracy[i,0] = seed[1]
	accuracy[i,1] = accuracy_score(y_test, predictions)
	# print("Accuracy: %.2f%%" % (accuracy * 100.0))
	'''
	model_2 = XGBClassifier()
	model_2.fit(X_train[:,[0,1]], y_train)
	y_pred = model_2.predict(X_test[:,[0,1]])
	predictions = [round(value) for value in y_pred]


	accuracy = accuracy_score(y_test, predictions)
	print("Accuracy: %.2f%%" % (accuracy * 100.0))

	model_2 = XGBClassifier()
	model_2.fit(X_train[:,[0,1,3,4,5,6,7,8]], y_train)
	y_pred = model_2.predict(X_test[:,[0,1,3,4,5,6,7,8]])
	predictions = [round(value) for value in y_pred]


	accuracy = accuracy_score(y_test, predictions)
	print("Accuracy: %.2f%%" % (accuracy * 100.0))

	model_2 = XGBClassifier()
	model_2.fit(X_train[:,[0,1,9,10]], y_train)
	y_pred = model_2.predict(X_test[:,[0,1,9,10]])
	predictions = [round(value) for value in y_pred]


	accuracy = accuracy_score(y_test, predictions)
	print("Accuracy: %.2f%%" % (accuracy * 100.0))

	model_2 = XGBClassifier()
	model_2.fit(X_train[:,[0,1,2,9,10]], y_train)
	y_pred = model_2.predict(X_test[:,[0,1,2,9,10]])
	predictions = [round(value) for value in y_pred]


	accuracy = accuracy_score(y_test, predictions)
	print("Accuracy: %.2f%%" % (accuracy * 100.0))
	'''
print("Accuracy: %.2f%%" % (np.mean(accuracy[:,1]) * 100.0))
print(accuracy)

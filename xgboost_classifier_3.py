from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import util
import numpy as np
import xgboost as xgb
# from xgboost import XGBClassifier
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
    X_new[count,addon+2] = np.ptp(i[i.nonzero()])
    '''if(temp == 0):
        X_new[count,addon+2] = np.max(X_new[:,addon+2])
    else:
        X_new[count,addon+2] = X_new[count,addon+0]*(1.0/(temp+0.00001))'''
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




model = xgb.XGBClassifier(missing=9999999999,
                max_depth = 7,
                n_estimators=700,
                learning_rate=0.1, 
                nthread=4,
                subsample=0.7,
                colsample_bytree=0.5,
                min_child_weight = 3,
                seed=1301)
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X_new[:,[6,7,8,9,10,11,12,13,14,15,16]], Y, cv=kfold)+0.05
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print(results.T)

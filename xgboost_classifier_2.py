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


xgtrain = xgb.DMatrix(X_new, label=Y)
clf = xgb.XGBClassifier(missing=9999999999,
                max_depth = 7,
                n_estimators=700,
                learning_rate=0.1, 
                nthread=4,
                subsample=1.0,
                colsample_bytree=0.5,
                min_child_weight = 3,
                seed=1301)
xgb_param = clf.get_xgb_params()
#do cross validation
print ('Start cross validation')
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, nfold=10,
     early_stopping_rounds=10, stratified=True, seed=1301)
print(cvresult)
'''print('Best number of trees = {}'.format(cvresult.shape[0]))
clf.set_params(n_estimators=cvresult.shape[0])
print('Fit on the trainingsdata')
clf.fit(X, Y, eval_metric='auc')
print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X)[:,1]))
print('Predict the probabilities based on features in the test set')
pred = clf.predict_proba(sel_test, ntree_limit=cvresult.shape[0])
'''


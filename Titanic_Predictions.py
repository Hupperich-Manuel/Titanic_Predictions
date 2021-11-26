import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


def Titanic_Predictions():
    
    df_train = pd.read_csv("train.csv", index_col=0)
    df_test = pd.read_csv("test.csv", index_col=0)
    gender_sub = pd.read_csv("gender_submission.csv", index_col=0)
    
    df_test = pd.merge(df_test, gender_sub, left_index=True, right_index=True)
    df_test['Sex'] = df_test['Sex'].replace("male", 0).replace("female", 1)
    df_train['Sex'] = df_train['Sex'].replace("male", 0).replace("female", 1)
    df_train.drop(['Name', 'Embarked', 'Ticket', 'Cabin'], 1, inplace=True)
    df_test.drop(['Name', 'Embarked', 'Ticket','Cabin'], 1, inplace=True)

    age = []

    for i in df_train['Age']:
        if str(i) == 'nan':
            age.append(df_train['Age'].mean())
        else:
            age.append(i)

    df_train['Age'] = age
    
    df_train = pd.get_dummies(df_train, columns=['Pclass'])
    df_test = pd.get_dummies(df_test, columns=['Pclass'])
    
    

    age = []

    for i in df_test['Age']:
        if str(i) == 'nan':
            age.append(df_test['Age'].mean())
        else:
            age.append(i)

    df_test['Age'] = age
    
    df_test.dropna(inplace=True)
    
    
    train_X, train_y = np.array(df_train[df_train.columns[1:]]),  np.array(df_train[df_train.columns[0]])
    test_X, test_Y =  np.array(df_test[df_test.columns[:-1]]), np.array(df_test[df_test.columns[-1:]])

    train_Y = np.expand_dims(train_y, axis=0)

    train_Y = train_Y.T
    
    
    pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA()),('model',LinearDiscriminantAnalysis())])

    params = {'model__solver':['svd']}

    search = GridSearchCV(pipe, param_grid=params, cv = 5, return_train_score=True)

    search.fit(train_X, train_Y)
    y_hat = search.predict(test_X)

    score = search.score(test_X, test_Y)
    #print("LR MSE score: %.2f // Score: %.02f"%(np.mean((y_hat-test_Y)**2), score))
    warnings.filterwarnings('ignore')
    
    return pd.DataFrame(y_hat, columns=['Survived'])

















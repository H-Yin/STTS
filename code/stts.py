import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import (
    LinearRegression,
    SGDRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

class STTS(object):
    
    def __init__(self, model, params=None):
        self.train_data = None
        self.test_data = None
        self.features = ['F1','F2','F3','F4','F5','F6',
                'F7','F8','F9','F10','F11','F12','F13','F14',
                'F15','F16','F17','F18','F19','F20','F21','F22']
        self.model = model
        self.params = params

    def load_train_data(self, path='../data/src/train.csv'):
        self.train_data = pd.read_csv(path, skiprows=[1], encoding='gb2312')

    def load_test_data(self, path='../data/src/test.csv'):
        self.test_data = pd.read_csv(path, skiprows=[1], encoding='gb2312')


    def train_one(self, target):
        X = self.train_data[self.features].values
        y = self.train_data[target].values
        train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=1024, test_size=0.25)
        model = self.model(**self.params)
        model.fit(train_x, train_y)
        pre_test_y = model.predict(test_x)
        mse = mean_squared_error(test_y, pre_test_y)
        score = 1.0 / (1 + mse)
        return (model, score)

    def train(self):
        self.load_train_data()
        target = ['A_Seqv', 'B_Seqv', 'C_Seqv', 'D_Seqv']
        scores  = []
        models = []
        for i in range(len(target)):
            (m, s) = self.train_one(target[i])
            models.append(m)
            scores.append(s)
        print(models[0].__class__.__name__, scores)
        return models

    def predict_one(self, model, target):
        test_x = self.test_data[self.features].values
        test_y = model.predict(test_x)
        res = pd.DataFrame(columns=['id','Seq_tag','Predict'])
        res['id'] = self.test_data['id']
        res['Seq_tag'] = pd.Series(np.full(self.test_data.shape[0], target.split('_')[0]))
        res['Predict'] = pd.Series(test_y)
        return res

    def predict(self, models):
        self.load_test_data()
        target = ['A_Seqv', 'B_Seqv', 'C_Seqv', 'D_Seqv']
        res = []
        for i  in range(len(target)):
            res.append(self.predict_one(models[i],target[i]))
        res = pd.concat(res)
        res.to_csv("../data/res/res_%s.csv" % models[0].__class__.__name__,
                encoding='utf-8', index=False)

if __name__=='__main__':
    models = [
        (LinearRegression, {}),
        (SGDRegressor, {}),
        #(SVR, {'kernel':'linear'}),
        #(SVR, {'kernel':'poly'}),
        #(SVR, {'kernel':'rbf'}),
        (KNeighborsRegressor, {'weights':'uniform'}),
        (KNeighborsRegressor, {'weights':'distance'}),
        (DecisionTreeRegressor, {}),
        (RandomForestRegressor, {}),
        (ExtraTreesRegressor, {}),
        (GradientBoostingRegressor, {}),
    ]
    for m in models:
        s= STTS(model=m[0], params=m[1])
        models = s.train()
        s.predict(models)

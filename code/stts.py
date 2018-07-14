import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    median_absolute_error,
    explained_variance_score,
    r2_score,
)
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
        self.features = ['F1','F2','F3','F4','F5','F6',
                'F7','F8','F9','F10','F11','F12','F13','F14',
                'F15','F16','F17','F18','F19','F20','F21','F22']
        
        self.train_data_path = '../data/src/train.csv'
        self.test_data_path = '../data/src/test.csv'

        self.train_data = None
        self.test_data = None
        
        self.model = model
        self.params = params
        
        self.target = ['A_Seqv', 'B_Seqv', 'C_Seqv', 'D_Seqv']
        
        self.o_headers = ['id','Seq_tag','Predict']

    def load_data(self):
        self.train_data = pd.read_csv(self.train_data_path, skiprows=[1], encoding='gb2312')
        self.train_data.corr().to_csv('../data/tmp/corr.csv')
        self.test_data = pd.read_csv(self.test_data_path, skiprows=[1], encoding='gb2312')

    def get_metrics(self, test_y, pre_test_y):
        res = []
        res.append(mean_squared_error(test_y, pre_test_y))
        res.append(mean_absolute_error(test_y, pre_test_y))
        res.append(explained_variance_score(test_y, pre_test_y))
        res.append(r2_score(test_y, pre_test_y))
        return res

    def train_one(self, target):
        X = self.train_data[self.features].values
        y = self.train_data[target].values
        train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=1024, test_size=0.25)
        model = self.model(**self.params)
        model.fit(train_x, train_y)
        pre_test_y = model.predict(test_x)
        metrics = self.get_metrics(test_y, pre_test_y)
        score = 1.0 / (1 + metrics[0])
        print(self.model.__name__, metrics, score)
        return (model, metrics, score)

    def train(self):
        scores  = []
        models = []
        for i in range(len(self.target)):
            (model, metrics, score) = self.train_one(self.target[i])
            models.append(model)
            scores.append(score)
            print(self.target[i], metrics)
        print(models[0].__class__.__name__, metrics, scores)
        return models

    def predict_one(self, model, target):
        test_x = self.test_data[self.features].values
        test_y = model.predict(test_x)
        res = pd.DataFrame(columns=self.o_headers)
        res['id'] = self.test_data['id']
        res['Seq_tag'] = pd.Series(np.full(self.test_data.shape[0], target.split('_')[0]))
        res['Predict'] = pd.Series(test_y)
        return res

    def predict(self, models):
        now = datetime.now().strftime("%Y%m%d%H%M")
        filename = "../data/res/res_%s_%s.csv" %(now, models[0].__class__.__name__)
        res = []
        for i  in range(len(self.target)):
            res.append(self.predict_one(models[i], self.target[i]))
        res = pd.concat(res)
        res.to_csv(filename, encoding='utf-8', index=False)

if __name__=='__main__':
    models = [
        #(LinearRegression, {}),
        #(SGDRegressor, {'max_iter':1000}),
        #(SVR, {'kernel':'linear'}),
        #(SVR, {'kernel':'poly'}),
        #(SVR, {'kernel':'rbf'}),
        #(KNeighborsRegressor, {'weights':'uniform'}),
        #(KNeighborsRegressor, {'weights':'distance'}),
        #(DecisionTreeRegressor, {}),
        #(RandomForestRegressor, {}),
        #(ExtraTreesRegressor, {}),
        (GradientBoostingRegressor, {}),
    ]
    for target in ['A_Seqv']:# 'B_Seqv', 'C_Seqv', 'D_Seqv']:
        print('-'*40, target, '-'*40)
        for m in models:
            s= STTS(model=m[0], params=m[1])
            s.load_data()
            model, metrics, score = s.train_one(target)
            s.predict_one(model, target)

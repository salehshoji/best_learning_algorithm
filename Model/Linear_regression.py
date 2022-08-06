import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, metrics
import Model.Load_data as ld
from sklearn.model_selection import train_test_split


class LinearRegression:

    def __init__(self, path):
        self.path = path
        # load the boston dataset
        self.df = ld.Load_data.load_csv(path)

    def data_provider(self):
        self.df.dtypes

    def asghar(self):
        columns = []
        for i in range(len(self.df.columns)):
            ttype = self.df.dtypes[i]
            if not (str(ttype).startswith("int") or str(ttype).startswith("float")):
                columns.append(self.df.columns[i])
        print(columns)
        df_encoded = pd.get_dummies(self.df, columns=columns)
        print(df_encoded.head())

    def calc(self):
        # boston = datasets.load_boston(return_X_y=False)

        # # defining feature matrix(X) and response vector(y)
        # X = boston.data
        # y = boston.target

        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]

        print(X.head())
        print("----------------")
        print(self.df.Breed1)
        self.df['Breed1'] = self.df.Breed1.astype('str').astype('int')
        print(self.df.Breed1)

        print(self.df.dtypes)

        # splitting X and y into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
        #                                                     random_state=1)
        #
        # # create linear regression object
        # reg = linear_model.LinearRegression()
        #
        # # train the model using the training sets
        # history = reg.fit(X_train, y_train)

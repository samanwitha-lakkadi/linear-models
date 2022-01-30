import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# TODO: Import all the linear regression models that you used here

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

# LOADING DATA
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
)

"""
# Wine-Quality Prediction with Linear Regression Models

Abstract: Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. The goal is to model wine quality based on physicochemical tests.
"""

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, sep=';', nrows=nrows)
    X = data.drop(['quality'], axis=1)
    T = data['quality']
    N = data.shape[0]
    # TODO: split training and testing to 80/20
    X_train, X_test, t_train, t_test = train_test_split(X,T,test_size = 0.2)
    return data, X_train, X_test, t_train, t_test

data, X_train, X_test, t_train, t_test = load_data(100000)


"## Summary"    
st.dataframe(data.describe())


#################### functions 
## TODO: copy and paste the evaluate and show_weights functions here to use 
# print the value text over the bar
# https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html

# evalutate function with above plots

def evaluate(y, t):
    plt.figure(figsize=(10,10))

    # t vs y plot
    plt.subplot(3,3, 1)
    # TODO: add the first plot
    plt.plot(t, y, '*')
    plt.plot([0,10], [0, 10], 'r--')
    plt.xlim([2.5, 8.5])
    plt.ylim([2.5, 8.5])
    plt.xlabel("target")
    plt.ylabel("predicted")
   
    # all value comparison
    plt.subplot(3,2, 2)
    # TODO: add the second one
    plt.plot(t, '.')
    plt.plot(y, 'x')
    plt.xlabel("samples")
    plt.ylabel("quality")
    
    # subplots of individual quality comparision
    # TODO: add the third subplots

    test = t

    plt.figure(figsize=(10,6))
    for k in range(3,9):
        plt.subplot(2,3, k-2)
        # TODO: add two plt.plot calls to produce scatter plot of GT and Pred
        plt.plot(test[np.where(test == k)],'.',label='GT')
        plt.plot(y[np.where(test == k)],'x',label='Pred')
    
        plt.ylim((2.5,8.5))
        plt.ylabel("quality")

    plt.legend()
    plt.tight_layout()
    
    
def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:0.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='blue')

def show_weights(model, names):
    
    # combine both the coefficients and intercept to present
    w = np.append(model.coef_, model.intercept_)


    plt.figure(figsize=(12,3))
    
    # TODO: create bar chart to present the w
    plt.bar(names,w)
    
    ax = plt.gca()
    ax.set_xticks(range(len(w)))
    ax.set_xticklabels(names, rotation = 90)
#     # manual positioning the label text - replaced with autolabel     
#     for i, v in enumerate(w):
#         ax.text(i - 0.25, v*1.05, "{:.2f}".format(v), color='red')
####################



# TODO: Add your codes to observe different models

print("Linear Regression")

lr_model = LinearRegression()
lr_model.fit(X_train, t_train)
lr_model.score(X_test, t_test)
y = lr_model.predict(X_test)
evaluate(y, t_test.to_numpy())
column_names = df.columns.values
show_weights(lr_model, column_names)


print("Ridge")

model = Ridge()
model.fit(X_train, t_train)
model.score(X_test, t_test)
y = model.predict(X_test)
evaluate(y,t_test.to_numpy())
show_weights(model, column_names)

print("Lasso")

model = Lasso()
model.fit(X_train, t_train)
model.score(X_test, t_test)
y = model.predict(X_test)
evaluate(y, t_test.to_numpy())
show_weights(model, column_names)

print("Elastic Net")


model = ElasticNet(random_state=0)
model.fit(X_train, t_train)
model.score(X_test, t_test)
y = model.predict(X_test)
evaluate(y , t_test.to_numpy())
show_weights(model, column_names)


print("Stochastic Gradient Descent")

model = SGDRegressor()
model.fit(X_train, t_train)
model.score(X_test, t_test)
y = model.predict(X_test)
evaluate(y , t_test.to_numpy())
show_weights(model, column_names)




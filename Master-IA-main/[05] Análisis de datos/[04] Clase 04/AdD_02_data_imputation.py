import numpy as np
from sklearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline




"""
Common vars
IMPUTER = ['constant', 'mean', 'knn', 'mice']
"""
PLOT_PAIR = False
NUMERICAL =['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
TARGET = ['Species']
TEST_SPLIT = .15
MISSING_METHOD = 'MCAR'
MISSING_RATIO = .8
IMPUTER = 'mean'
TRUNC_CENS = 'truncated'




"""
Read data
"""
dfs = []
folder = './data'
for i in range(5):
    file = 'df' + str(i) + '.csv'
    dfs.append(pd.read_csv(folder + '/' + file))




"""
Split data
"""
X_trains = []
X_tests  = []
y_trains = []
y_tests  = []
for df in dfs:
    X = df[NUMERICAL] 
    y = df[TARGET]
    (X_train,
     X_test,
     y_train,
     y_test) = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42)
    X_trains.append(X_train)
    X_tests .append(X_test)
    y_trains.append(y_train)
    y_tests .append(y_test)




"""
Classification
"""
results0 = []
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=42)
#clf = SVC(kernel="linear", C=0.025, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
for X_train, y_train, X_test, y_test in zip(X_trains, y_trains, X_tests, y_tests):
    #clf.fit(X=X_train, y=y_train.squeeze())
    #score = cross_val_score(clf, X_test, y_test.squeeze(), scoring='accuracy', cv=3)
    score = cross_val_score(clf, X_train, y_train.squeeze(), scoring= 'accuracy' , cv=cv)
    results0.append(score)

    print(f'Score =  {np.mean(score):5.3f}')




"""
Data elimination
"""
X_trains_missing = []
# MCAR
if MISSING_METHOD == 'MCAR':
    for X_train, y_train in zip(X_trains, y_trains):
        mask = np.random.choice(a=[np.nan, 1.0], size=X_train.shape, p=[MISSING_RATIO,1-MISSING_RATIO])
        X_train_aux = X_train * mask
        X_trains_missing.append(X_train_aux)
# MAR
if MISSING_METHOD == 'MAR':
    for X_train, y_train in zip(X_trains, y_trains):
        X_train_aux = X_train.copy()
        rows = np.array(X_train_aux.index)
        cols = np.array(X_train_aux.columns)
        for i in range(1000):
            row = np.random.choice(rows)
            col = np.random.choice(cols)
            val = X_train_aux.loc[row, col]
            aux_cols = list(copy(cols))
            aux_cols.remove(col)
            aux_col = np.random.choice(aux_cols)
            aux_val = X_train_aux.loc[row, aux_col]
            if not np.isnan(val):
                mean = X_train_aux[aux_col].mean()
                std = X_train_aux[aux_col].std()
                if TRUNC_CENS == 'truncated':
                    if aux_val > mean+std*2: # Truncado
                        X_train_aux.loc[row, col] = np.nan
                if TRUNC_CENS == 'censored':
                    if mean+std < val < mean+std: # Censurado
                        X_train_aux.loc[row, col] = np.nan
            if X_train_aux.isna().sum().sum()/len(X_train_aux) >= MISSING_RATIO:
                break
        X_trains_missing.append(X_train_aux)
# MNAR
if MISSING_METHOD == 'MNAR':
    for X_train, y_train in zip(X_trains, y_trains):
        X_train_aux = X_train.copy()
        rows = np.array(X_train_aux.index)
        cols = np.array(X_train_aux.columns)
        for i in range(1000):
            row = np.random.choice(rows)
            col = np.random.choice(cols)
            val = X_train_aux.loc[row, col]
            if not np.isnan(val):
                mean = X_train_aux[col].mean()
                std = X_train_aux[col].std()
                if TRUNC_CENS == 'truncated':
                    if val > mean+std*2: # Truncado
                        X_train_aux.loc[row, col] = np.nan
                if TRUNC_CENS == 'censored':
                    if mean+std < val < mean+std: # Censurado
                        X_train_aux.loc[row, col] = np.nan
            if X_train_aux.isna().sum().sum()/len(X_train_aux) >= MISSING_RATIO:
                break
        X_trains_missing.append(X_train_aux)




"""
Data imputation and classification
"""
for i,(X_train_missing, X_train, y_train, X_test, y_test) in enumerate(zip(X_trains_missing, X_trains, y_trains, X_tests, y_tests)):
    print(f'Dataset {i}')
    results = []
    mses = []

    imputer_list = ['constant', 'mean', 'knn', 'mice']
    imputer_idx  = imputer_list.index(IMPUTER)
    param_grid = [
        [None,],
        [None,],
        [1,3,5,7,9,11,],
        [None,2,1,],
    ]

    for param in param_grid[imputer_idx]:
        if imputer_idx == 0:
            imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=10.0)
        if imputer_idx == 1:
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        if imputer_idx == 2:
            imputer = KNNImputer(n_neighbors=param)
        if imputer_idx == 3:
            imputer = IterativeImputer(n_nearest_features=param)
        clf = DecisionTreeClassifier(random_state=42)
        X_train_imputed = imputer.fit_transform(X_train_missing)
        mse = np.mean(np.array(X_train - X_train_imputed)**2)
        pipeline = Pipeline(steps=[
            ( 'imputer', imputer),
            ( 'clf'    , clf)
        ])

        #scores = cross_val_score(pipeline, X_train_missing, y_train.squeeze(), scoring='accuracy', cv=3)
        scores = cross_val_score(pipeline, X_train_missing, y_train.squeeze(), scoring='accuracy' , cv=cv)
        results.append(scores)
        mses.append(mse)
        print(f'MSE={mse:8.3f}, Score={np.mean(scores):6.3f} ({np.mean(results0[i]):6.3f})')

    plt.figure()
    plt.title(f'Dataset {i}')
    plt.boxplot(results, showmeans=True)
    plt.grid()
    plt.show()
    plt.close()

    if PLOT_PAIR:
        df_aux = pd.DataFrame(np.concatenate((X_train_imputed, y_train,), axis=1))
        columns = [ 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species', ]
        df_aux.columns=columns
        sns.pairplot(data=df_aux, hue='Species'); plt.show()
        plt.show()
        plt.close()












































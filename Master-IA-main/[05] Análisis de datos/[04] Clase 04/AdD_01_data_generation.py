import numpy as np
from sklearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




PLOT = True


"""
Common vars
"""
sd_factor = 1
mu_factor = 3
sd_min = 0.3
mu_min = 3
mu = np.array([[1,2,3,4],[5,6,6,3],[1,3,5,-2]])
sd = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
coef = [0.3, 1.2, 3,]
n_samples = 150
targets = np.floor(np.linspace(0,3-1/n_samples,n_samples))[...,None]
columns = [
        'SepalLengthCm',
        'SepalWidthCm',
        'PetalLengthCm',
        'PetalWidthCm',
        'Species',
        ]




"""
Dataset 0: Random
"""
#mu = [1, 4, -3, 0]
#sd = [1, 2, 2, 5]
mu0 = np.random.rand(4) * mu_factor + mu_min
sd0 = np.random.rand(4) * sd_factor + sd_min
d  = []
d.append(sd0[0] * np.random.randn(n_samples,1) + mu0[0])
d.append(sd0[1] * np.random.randn(n_samples,1) + mu0[1])
d.append(sd0[2] * np.random.randn(n_samples,1) + mu0[2])
d.append(sd0[3] * np.random.rand(n_samples,1)  + mu0[3])
data = np.concatenate((d[0],d[1],d[2],d[3],targets,), axis=1)
df0 = pd.DataFrame(data)




"""
Dataset 1: Linear independent
"""
#mu = [[1, 4, -3, 0],
#      [3, 3, -4, 1],
#      [2, 3, -1, 0],]
#sd = [1, 2, 2, 5]
#mu = np.random.rand(3,4) * mu_factor + mu_min
#sd = np.random.rand(3,4) * sd_factor + sd_min
#mu = np.array([[1,2,3,4],[5,6,7,8],[1,3,5,7]])
#sd = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
d  = []
d.append(np.zeros((n_samples, 1,)) * np.nan)
d.append(np.zeros((n_samples, 1,)) * np.nan)
d.append(np.zeros((n_samples, 1,)) * np.nan)
d.append(np.zeros((n_samples, 1,)) * np.nan)
for i,t in enumerate(np.unique(targets)):
    for j in range(len(d)):
        mask = np.array([targets==t]).ravel()
        len_ = np.sum(mask)
        d[j][mask] = sd[i,j] * np.random.randn(len_,1) + mu[i,j]
data = np.concatenate((d[0],d[1],d[2],d[3],targets,), axis=1)
df1 = pd.DataFrame(data)



"""
Dataset 2: Linear dependent
"""
#mu = np.random.rand(3,4) * mu_factor + mu_min
#sd = np.random.rand(3,4) * sd_factor + sd_min
#mu = np.array([[1,2,3,4],[5,6,7,8],[1,3,5,7]])
#sd = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
#coef = np.random.rand(3,)
d  = []
d.append(np.zeros((n_samples, 1,)) * np.nan)
d.append(np.zeros((n_samples, 1,)) * np.nan)
d.append(np.zeros((n_samples, 1,)) * np.nan)
d.append(np.zeros((n_samples, 1,)) * np.nan)
# Independent vars
for i,t in enumerate(np.unique(targets)):
    for j in range(len(d)-1):
        mask = np.array([targets==t]).ravel()
        len_ = np.sum(mask)
        d[j][mask] = sd[i,j] * np.random.randn(len_,1) + mu[i,j]
# Dependent var
d[-1] = np.zeros((n_samples, 1,))
for i in range(len(d)-1):
    d[-1] += d[i] * coef[i]
data = np.concatenate((d[0],d[1],d[2],d[3],targets,), axis=1)
df2 = pd.DataFrame(data)




"""
Dataset 3: Non-linear dependent
"""
#mu = np.random.rand(3,4) * mu_factor + mu_min
#sd = np.random.rand(3,4) * sd_factor + sd_min
#mu = np.array([[1,2,3,4],[5,6,7,8],[1,3,5,7]])
#sd = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
#coef = [0.3, 1.2, 3,]
d  = []
d.append(np.zeros((n_samples, 1,)) * np.nan)
d.append(np.zeros((n_samples, 1,)) * np.nan)
d.append(np.zeros((n_samples, 1,)) * np.nan)
d.append(np.zeros((n_samples, 1,)) * np.nan)
# Independent vars
for i,t in enumerate(np.unique(targets)):
    for j in range(len(d)-2):
        mask = np.array([targets==t]).ravel()
        len_ = np.sum(mask)
        d[j][mask] = sd[i,j] * np.random.randn(len_,1) + mu[i,j]
# Dependent var
d[-1] = np.array(d[0])**2 + sd[0,2] * np.random.randn(n_samples,1) + mu[0,2]
d[-2] = 3/(np.abs(d[1]-np.mean(d[1])+0.2))  + sd[0,3] * np.random.randn(n_samples,1) + mu[0,3]
#d[-2] = np.exp(np.abs(d[1])) + sd[0,3] * np.random.randn(n_samples,1) + mu[0,3]
data = np.concatenate((d[0],d[1],d[2],d[3],targets,), axis=1)
df3 = pd.DataFrame(data)




"""
Dataset 4: Iris dataset
"""
iris = datasets.load_iris()
df_iris = np.concatenate((iris.data, iris.target[...,None],), axis=1)
df_iris = pd.DataFrame(df_iris)




"""
Print datasets
"""
df = [df0, df1, df2, df3, df_iris,]
for v in df:
    v.columns = columns
for v in df:
    print(v.head())
if PLOT:
    for i,v in enumerate(df):
        g = sns.pairplot(data=v, hue='Species')
        g.fig.suptitle(f'Dataset {i}')
        plt.show()
        plt.close()
folder = './data'
for i,v in enumerate(df):
    file = 'df' + str(i) + '.csv'
    v.to_csv(folder + '/' + file, index=False)



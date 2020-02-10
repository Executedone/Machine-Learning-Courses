```python
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
```


```python
data10_1 = sio.loadmat('./ex8_movies.mat')
data10_2 = sio.loadmat('./ex8_movieParams')
Y,R = data10_1['Y'],data10_1['R']
X,Theta = data10_2['X'],data10_2['Theta']
```


```python
#loss function
def loss_cofi(X,theta,lamda=0):
    loss = 0.5 * np.sum((X.dot(theta.T) * R - Y) ** 2)
    reg = 0.5 * lamda * (np.sum(theta ** 2) + np.sum(X ** 2))
    return loss + reg

#梯度下降
def grad_cofi(X,theta,lamda=0):
    X_temp = X - alpha * ((X.dot(theta.T) * R - Y).dot(theta) + lamda * X)
    theta_temp = theta - alpha * (np.dot(X.T,X.dot(theta.T) * R - Y).T + lamda * theta)
    return X_temp,theta_temp
```


```python
#collaborative filtering,X,Theta
#初始化变量，学习率，迭代次数
# x,theta = X,Theta
for j in range(100):
    iteration = 300
# Loss = []
    for i in range(iteration):
        alpha = 0.2 / (i+200)
        loss = loss_cofi(x,theta)
        x,theta = grad_cofi(x,theta)
#     Loss.append(loss)
# plt.scatter(range(iteration),Loss,c='b')
# plt.show()
loss
```




    22791.447002242898




```python
predict = x.dot(theta.T)
```


```python

```


```python

```

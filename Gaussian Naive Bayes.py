>>> import numpy as np

>>> #生成可利用的训练点
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> Y = np.array([1, 1, 1, 2, 2, 2])

>>> from sklearn.naive_bayes import GaussianNB #调用外部函数
#创建分类器 Classifier
>>> clf = GaussianNB()
>>> clf.fit(X, Y) #fit 拟合，也就是train 训练函数，x为feature，y为labels
GaussianNB(priors=None, var_smoothing=1e-09)

#让分类器做预测
>>> print(clf.predict([[-0.8, -1]]))
[1]


>>> clf_pf = GaussianNB()
>>> clf_pf.partial_fit(X, Y, np.unique(Y))
GaussianNB(priors=None, var_smoothing=1e-09)
>>> print(clf_pf.predict([[-0.8, -1]]))

# Day_03_09_leaf.py
import numpy as np
import pandas as pd
from sklearn import (model_selection, neighbors, preprocessing, svm, linear_model)

leaf = pd.read_csv('Data/leaf.csv')
print(leaf)

x = leaf.drop(labels=['id', 'species'], axis=1)     # drop은 1번이 열임
y = leaf.species
print(x.shape, y.shape)
# (990, 192) (990,)
print(y[:5])    # 적합한 형태가 아님
# 0              Acer_Opalus
# 1    Pterocarya_Stenoptera
# 2     Quercus_Hartwissiana
# 3          Tilia_Tomentosa
# 4       Quercus_Variabilis
# Name: species, dtype: object


y = preprocessing.LabelEncoder().fit_transform(y)
print(y[:5])
# 0              Acer_Opalus
# 1    Pterocarya_Stenoptera
# 2     Quercus_Hartwissiana
# 3          Tilia_Tomentosa
# 4       Quercus_Variabilis
# Name: species, dtype: object
# [ 3 49 65 94 84]
#
data = model_selection.train_test_split(x,y)
x_train, x_text, y_train, y_test = data

classifier = [svm.SVC(),
              neighbors.KNeighborsClassifier(),
              linear_model.LogisticRegression()]
# 다양한 경우에 대해서 해보고 좋은 것을 찾아서 이것을 집중적으로 연구


for clf in classifier:
    # clf = svm.SVC()
    clf.fit(x_train, y_train)

    print('train :', clf.score(x_train, y_train))
    print('test :', clf.score(x_text, y_test))










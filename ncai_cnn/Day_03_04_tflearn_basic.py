# Day_03_04_tflearn_basic.py
import numpy as np
import tflearn

tflearn.datasets.titanic.download_dataset('Data/titanic.csv')

passengers, labels = tflearn.data_utils.load_csv(
    'Data/titanic.csv', target_column=0,
    categorical_labels=True, n_classes=2)

print(passengers)
print(passengers[0])
print(type(passengers))
# ['1', 'Allen, Miss. Elisabeth Walton', 'female', '29', '0', '0', '24160', '211.3375']
# <class 'list'>

print(labels[0])            # [0. 1.]
print(len(passengers))      # 1309
print('-' * 50)

for item in passengers:
    item[2] = int(item[2] == 'female')
    item.pop(6)
    item.pop(1)

    print(item)

passengers = np.float32(passengers)
print(passengers[0])
print('-' * 50)

net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(passengers, labels, n_epoch=10,
          batch_size=16, show_metric=True)

dicaprio = [3, 0, 19, 0, 0, 5]
winslet = [1, 1, 17, 1, 2, 100]

pred = model.predict([dicaprio, winslet])
print('Dicaprio :', pred[0][1])
print('Winslet  :', pred[1][1])
print(pred)




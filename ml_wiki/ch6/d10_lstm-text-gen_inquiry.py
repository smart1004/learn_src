import codecs
from bs4 import BeautifulSoup
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random, sys
from keras.models import model_from_json
import os
'''
# keras model save
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/  '''

text = codecs.open("C:/home2/ml_wiki/ch6/inquiry.csv", "r", encoding="euc_kr").read()

def getModel():
    dir1 = "c:/home2/"
    json_file_name = "model_inquiry.json"

    file_path = dir1 + "/" + json_file_name
    if not os.path.exists(file_path):
        return create_model()
    else:
        return exist_model()

def create_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def exist_model():
    # load json and create model
    json_file = open('model_inquiry.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_inquiry.h5")
    optimizer = RMSprop(lr=0.01)
    #loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    print("Loaded model from disk")
    return loaded_model

print('코퍼스의 길이: ', len(text))
# 문자를 하나하나 읽어 들이고 ID 붙이기
chars = sorted(list(set(text))) # set
print('사용되고 있는 문자의 수:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars)) # 문자 → ID
indices_char = dict((i, c) for i, c in enumerate(chars)) # ID → 문자
# 텍스트를 maxlen개의 문자로 자르고 다음에 오는 문자 등록하기
maxlen = 20
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('학습할 구문의 수:', len(sentences))
print('텍스트를 ID 벡터로 변환합니다...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
# 모델 구축하기(LSTM)
print('모델을 구축합니다...')
model = getModel()

# 후보를 배열에서 꺼내기
def sample(preds, temperature=1.0):  # sample(preds, diversity)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# 학습시키고 텍스트 생성하기 반복
for iteration in range(1, 5):
    print()
    print('-' * 50)
    print('반복 =', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1) #

    # 임의의 시작 텍스트 선택하기
    start_index = random.randint(0, len(text) - maxlen - 1)
    # 다양한 다양성의 문장 생성
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('--- 다양성 = ', diversity)
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('--- 시드 = "' + sentence + '"')
        sys.stdout.write(generated)

        # 시드를 기반으로 텍스트 자동 생성
        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.
            # 다음에 올 문자를 예측하기
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            # 출력하기
            generated += next_char
            sentence  = sentence[1:] + next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

    # serialize model to JSON
    model_json = model.to_json()
    with open("model_inquiry.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_inquiry.h5")
    print("Saved model to disk")

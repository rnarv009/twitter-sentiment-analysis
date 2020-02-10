import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def read_data():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    # train = train.head(10)
    # test = test.head(10)
    print("train size:", train.shape)
    print('test size:', test.shape)
    return train, test


train, test = read_data()


def preprocess_data(vec):
    vec = vec.replace('[', '').replace(']', '')
    vec = vec.split(', ')
    vec = [float(j) for j in vec]
    return vec


def create_train_val_data(train, test):
    train['tweet'] = train['tweet'].apply(lambda a: preprocess_data(a))
    test['tweet'] = test['tweet'].apply(lambda a: preprocess_data(a))

    X = train['tweet']
    Y = train['label']
    X = [np.array(i) for i in X]
    X = np.array(X)
    Y = np.array(list(Y))
    X_test = test['tweet']
    X_test = [np.array(i) for i in X_test]
    X_test = np.array(X_test)

    X_tr, X_val, Y_tr, Y_val = train_test_split(X, Y, test_size=0.33, random_state=42)

    print(X_tr.shape)
    print(Y_tr.shape)
    print(X_val.shape)
    print(Y_val.shape)
    print(X_test.shape)

    return X, Y, X_tr, Y_tr, X_val, Y_val, X_test


X, Y, X_tr, Y_tr, X_val, Y_val, X_test = create_train_val_data(train, test)

# # create models
# models = {'knn': KNeighborsClassifier(), 'logit': LogisticRegression(), 'svm': SVC()}
#
# for model, call in models.items():
#     print(f'Training {model}')
#     call.fit(X_tr, Y_tr)
#     pred = call.predict(X_val)
#     print(accuracy_score(pred, Y_val))
#     print()


model = SVC()
model.fit(X, Y)
y_pred = model.predict(X_test)
df = pd.DataFrame()
df['id'] = test['id']
df['label'] = y_pred
df.to_csv('../data/submission.csv', index=False)

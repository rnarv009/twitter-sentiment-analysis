import pandas as pd
# import numpy as np
import re
from bert_serving.client import BertClient

vectorizer = BertClient(ip='localhost', port=8190)
e = vectorizer.encode(['rahul'])


# print(e)
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    return text


train = pd.read_csv('../data/train_E6oV3lV.csv')
train=train.head(5)
print(train.shape)

test = pd.read_csv('../data/test_tweets_anuFYb8.csv')
test = test.head(5)
print(test.shape)

sample = pd.read_csv('../data/sample_submission_gfvA5FD.csv')
print(sample.head())

tr_tweets = train['tweet'].to_list()
tr_tweets = [preprocess(i) for i in tr_tweets]
x_train = vectorizer.encode(tr_tweets)
# print("length of training data is:", type(x_train))
# print("vector for training:", type(x_train[0]))
# print("training data:", x_train[0][0])
x_train = [list(i) for i in x_train]

df1 = pd.DataFrame()
df1['id'] = train['id'].to_list()
print(df1.head())
df1['tweet'] = x_train
df1['label'] = train['label']
# df1.to_csv('../data/train.csv', index=False)


te_tweets = test['tweet'].to_list()
te_tweets = [preprocess(i) for i in te_tweets]
x_test = vectorizer.encode(te_tweets)
print("length of testing data is :", x_test.shape)
x_test = [list(i) for i in x_test]
df2 = pd.DataFrame()
df2['id'] = test['id'].to_list()
df2['tweet'] = x_test
# df2.to_csv('../data/test.csv', index=False)

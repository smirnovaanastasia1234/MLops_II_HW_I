import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pickle
import sys
import os
import yaml

#Загрузка данных
f_input = sys.argv[1]
f_output = os.path.join("models", sys.argv[2])
os.makedirs(os.path.join("models"), exist_ok=True)

params = yaml.safe_load(open("params.yaml"))["split"]
p_split_ratio = params["split_ratio"]
params1 = yaml.safe_load(open("params.yaml"))["train"]
p_seed = params1["seed"]
p_max_iter = params1["max_iter"]

df = pd.read_csv(f_input)
print(df.shape)
print(df.head())
df.Predicted.value_counts()
#Разделение данных на функции и метки
X = df[['url']].copy()
y = df.Predicted.copy()
#Обработка данных
tokenizer = RegexpTokenizer(r'[A-Za-z]+') #[a-zA-Z]обозначает один символ от a до z или от A доZ
stemmer = SnowballStemmer("english")
cv = CountVectorizer()
def prepare_data(X) :
    X['text_tokenized'] = X.url.map(lambda t: tokenizer.tokenize(t)) #Разделение на токены
    X['text_stemmed'] = X.text_tokenized.map(lambda t: [stemmer.stem(word) for word in t])#stemmer приводит слова с одним корнем к одному слову
    X['text_sent'] = X.text_stemmed.map(lambda t: ' '.join(t)) #Объеденяем список в предложение
    features = cv.fit_transform(X.text_sent)
    return X, features
X, features = prepare_data(X)
#Обучение модели
logreg = LogisticRegression(max_iter=p_max_iter)
trainX, testX, trainY, testY = train_test_split(features, y, test_size=p_split_ratio, stratify=y, random_state=p_seed)
logreg.fit(trainX, trainY)

#Сохранение обученной модели
with open(f_output, 'wb') as output:
    pickle.dump(logreg, output)

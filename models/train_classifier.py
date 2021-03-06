import sys
import numpy as np
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,precision_score, recall_score, f1_score

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    ''' 加载数据
    A'r'g's:
        database_filepath
    Returns:
        X: pd.Series, 灾难消息序列
        y: df, 消息的灾难类别标注情况
        category_names:list, 灾难类别名称
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(y.columns)
    return X,y,category_names

def tokenize(text): 
    ''' 提取文本关键词
    Args:
        text: string, 一条灾难消息文本
    Returns:
        clean_tokens: list, 文本清洗+token提取
    '''
    url_pat = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    urls = re.findall(url_pat, text)
    
    for url in urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    stops = list(set(stopwords.words('english')))
    stops += ["http","&","-",":",",",".","(",")","#"]

    clean_tokens = [token for token in clean_tokens if token not in stops]

    return clean_tokens

def build_model():
    ''' 模型训练
    Args:
        无
    Returns:
        model: 创建的模型
    '''
    model = Pipeline([ 
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf',TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
                ])

    parameters = {
        #'clf__estimator__max_features': ['log2'],
        'clf__estimator__n_estimators':[50]
        }
    model = GridSearchCV(estimator=model, param_grid=parameters, cv=5, n_jobs=1)
    return model


def evaluate_model(model, X_test, y_test, category_names):
    '''模型评估
    Args:
        model: 创建的模型
        X_test: 测试集X
        y_test: 测试集y
        category_names: 灾难类别名称
    Returns:
        scores: df, 包括：每个灾难类别预测的 精确度，召回率，f1分数，真实值y的分布，预测值y的分布
    '''
    y_pred = model.predict(X_test)

    scores = []
    for j in range(len(y_test.columns)):
        scores.append([precision_score(y_test.iloc[:, j].values, y_pred[:, j], average='weighted'),
                        recall_score(y_test.iloc[:, j].values, y_pred[:, j], average='weighted'),
                        f1_score(y_test.iloc[:, j].values, y_pred[:, j], average='weighted'),
                        Counter(y_test.iloc[:,j]),
                        Counter(y_pred[:,j]) 
                        ])
    scores = pd.DataFrame(scores, columns=['precision','recall_score','f1_score','y_test','y_pred'],
                         index=category_names)
    print(round(scores,3))
    return scores


def save_model(model, model_filepath):
    ''' 保存模型以备后续接口加载使用
    Args:
        model:训练好的模型
        model_filepath: 要存入的文件路径
    Returns:
        无，生成了一个训练好的模型文件
    '''
    with open(model_filepath,'wb') as fw:
        pickle.dump(model,fw)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    main()
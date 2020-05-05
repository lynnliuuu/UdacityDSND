# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    ''' 加载灾害消息数据文件
    Args:
        messages_filepat (csv): 灾害消息文件
        categories_filepath (csv): 灾害消息分类标注文件
    Returns:
        df（pandas df）: 两个文件根据id合并成一个df
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages,categories, on='id')
    return df

def clean_data(df):
    ''' 清洗灾害消息数据
    Args:
        df（pandas df）: 原始灾害消息数据
    Returns:
        df（pandas df）: 清洗后的灾害消息数据，36类标注全部清洗为0/1变量
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    row = categories.head(1)
    category_colnames = row.apply(lambda x: x.str.split("-")[0][0], axis=0).values
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # set each value to be the last character of the string
    for column in categories:
        categories[column] = categories[column].str.split("-",expand=True)[1]
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()

    # raplace 2 with 1
    df.related.replace(2,1,inplace=True)
    
    return df

def save_data(df, database_filename):
    ''' 将灾害数据存入数据库中
    Args:
        df（pandas df）: 原始灾害消息数据
        database_filename: 指定的数据库文件路径
    Returns:
        无；创建了数据库，并向指定数据库中写入了表：DisasterResponse
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    print(database_filename)
    df.to_sql('DisasterResponse', engine, index=False,if_exists='replace')
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
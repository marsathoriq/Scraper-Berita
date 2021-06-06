import os
import re
import string
import joblib
import operator
import numpy as np
import tensorflow as tf
import pandas as pd
import sqlalchemy

from google.cloud import storage
from datetime import datetime, timedelta
from detik import get_detik_dataframe_from_date
from kompas import get_kompas_dataframe_from_date
from database import init_connection_engine


## Download model
def download_model(bucket_name, source_dir, destination_folder, model_version):
    download_directory = os.path.join(destination_folder, "models", model_version)
    variables_directory = os.path.join(destination_folder, "models", model_version, "variables")
    if not os.path.exists(download_directory):
        os.makedirs(download_directory)

    if not os.path.exists(variables_directory):
        os.makedirs(variables_directory) 

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    sub_folders = bucket.list_blobs(prefix=source_dir)

    for file in sub_folders:
        download_path = os.path.join(download_directory, os.path.basename(file.name))
        if "variables" in file.name:
            download_path = os.path.join(variables_directory, os.path.basename(file.name))
        file.download_to_filename(download_path)

    print('Model downloaded to {}.'.format(download_directory))
    return download_directory

## Download blob
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

## Convert dict to str
def get_content(data):
  return " ".join(list(data.values()))

def check_len(data):
  return len(data.split())

# bersihkan teks
def clean_text(text, stop_words):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = str(text)
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\r', ' ', text)
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return text

def vectorize_verif(data, CountVectorizer):
    bow_data = CountVectorizer.transform(data['content'])
    bow_data = pd.DataFrame.sparse.from_spmatrix(bow_data,columns=CountVectorizer.get_feature_names())
    return bow_data

def predict_data(data, CountVectorizer, stop_words, model):
    data['content'] = data['content'].apply(clean_text, stop_words=stop_words)
    bow_data = vectorize_verif(data, CountVectorizer)
    return model.predict(bow_data)

def change_label(value):
    if value > 0.6:
        return 1
    elif value < 0.4:
        return -1
    else:
        return 0

def full_predict(event, context):
    yesterday_date = datetime.now() - timedelta(1)
    date_now = datetime.strftime(yesterday_date, '%d/%m/%Y')
    if isinstance(event, dict) and not(event.get('attributes') is None) :
        date_now = event.get('attributes').get('date')
    print(date_now)
    print(event)

    datetime_obj_now = datetime.strptime(date_now, '%d/%m/%Y')
    date_sql_now = datetime_obj_now.strftime('%Y-%m-%d')

    ## Get news
    detik_news = get_detik_dataframe_from_date(date_now)
    kompas_news = get_kompas_dataframe_from_date(date_now)
    news_data = pd.concat([detik_news, kompas_news]).reset_index(drop=True)

    ## Preprocess
    news_data['content'] = detik_news['content'].apply(get_content)
    # news_data['Len'] = news_data['content'].apply(check_len)
    # news_data = news_data[news_data['Len'] > 10].reset_index(drop=True)

    ## Load tensorflow model for sentiment analysis
    dest_folder = '/tmp/'
    download_dir_sentiment = download_model(
        bucket_name = os.environ['BUCKET_NAME'],
        source_dir = 'Sentiment/model',
        destination_folder = dest_folder,
        model_version = '1'
    )

    dest_vectorizer_sentiment = os.path.join(dest_folder, 'CountVectorizerSentiment.pkl')
    download_blob(
        bucket_name = os.environ['BUCKET_NAME'],
        source_blob_name = 'Sentiment/CountVectorizer.pkl',
        destination_file_name = dest_vectorizer_sentiment
    )

    ## Load tf model for topic classification
    download_dir_jakarta = download_model(
        bucket_name = os.environ['BUCKET_NAME'],
        source_dir = 'Jakarta/model',
        destination_folder = dest_folder,
        model_version = '2'
    )

    dest_vectorizer_jakarta = os.path.join(dest_folder, 'CountVectorizerJakarta.pkl')
    download_blob(
        bucket_name = os.environ['BUCKET_NAME'],
        source_blob_name = 'Jakarta/CountVectorizer.pkl',
        destination_file_name = dest_vectorizer_jakarta
    )

    ## Load tf model for text classification
    download_dir_topic = download_model(
        bucket_name = os.environ['BUCKET_NAME'],
        source_dir = 'Topic/model',
        destination_folder = dest_folder,
        model_version = '3'
    )

    dest_vectorizer_topic = os.path.join(dest_folder, 'CountVectorizerTopic.pkl')
    download_blob(
        bucket_name = os.environ['BUCKET_NAME'],
        source_blob_name = 'Topic/CountVectorizer.pkl',
        destination_file_name = dest_vectorizer_topic
    )

    ## Predict jakarta
    CountVectorizer = joblib.load(dest_vectorizer_jakarta)
    stop_words = open("stopword.txt", "r").read().split()
    model = tf.keras.models.load_model(download_dir_jakarta)

    pred = predict_data(news_data, CountVectorizer, stop_words, model).round()
    news_data['jakarta'] = pred

    ## Predict topic
    CountVectorizer = joblib.load(dest_vectorizer_topic)
    stop_words = open("stopword.txt", "r").read().split()
    model = tf.keras.models.load_model(download_dir_topic)

    pred = predict_data(news_data, CountVectorizer, stop_words, model)
    news_data['topic'] = np.argmax(pred, axis=1)

    ## Predict sentiment
    CountVectorizer = joblib.load(dest_vectorizer_sentiment)  
    model = tf.keras.models.load_model(download_dir_sentiment)

    pred = predict_data(news_data, CountVectorizer, stop_words, model)

    vchangeLabel = np.vectorize(change_label)
    news_data['sentiment'] = vchangeLabel(pred)
    
    print(news_data)

    # Filter news_data that only related to jakarta government
    news_data = news_data[news_data['jakarta'] == 1]

    data_positif = news_data[news_data['sentiment'] == 1].groupby('topic')
    data_negatif = news_data[news_data['sentiment'] == -1].groupby('topic')

    cnt_tags_positive = {}
    for topic, group in data_positif:
        value = {}
        for row in group.itertuples():
            for tag in row.tags:
                value[tag] = value.get(tag, 0) + 1 
        
        cnt_tags_positive[topic] = value

    cnt_tags_negative = {}
    for topic, group in data_negatif:
        value = {}
        for row in group.itertuples():
            for tag in row.tags:
                value[tag] = value.get(tag, 0) + 1 
        
        cnt_tags_negative[topic] = value
    
    # sorted_cnt_positive = sorted(cnt_tags_positive.items(), key=operator.itemgetter(1),reverse=True)
    # sorted_cnt_negative = sorted(cnt_tags_negative.items(), key=operator.itemgetter(1),reverse=True)
    
    # print(sorted_cnt_positive)
    # print(sorted_cnt_negative)

    ## Insert to portal table
    db = init_connection_engine()
    with db.connect() as conn:
        for _, row in news_data.iterrows():
            stmt = sqlalchemy.text(
                "INSERT INTO portals (portal, date, link, title, image, content, sentiment, id_category) "
                "VALUES(:portal, :date, :link, :title, :image, :content, :sentiment, :id_category)"
            )
            result = conn.execute(
                stmt,
                portal = row['portal'],
                date = date_sql_now,
                link = row['link'],
                title = row['title'],
                image = row['image'],
                content = row['content'],
                sentiment = row['sentiment'],
                id_category = row['topic']
            )
            portal_id = result.lastrowid
            for tag in row['tags']:
                stmt_tag = sqlalchemy.text(
                    "INSERT INTO tags (id_portal, tag) "
                    "VALUES(:id_portal, :tag)"
                )
                conn.execute(
                    stmt_tag,
                    id_portal = portal_id,
                    tag = tag
                )
        
        for category in cnt_tags_positive:
            tags = cnt_tags_positive.get(category)
            for tag in tags:
                stmt = sqlalchemy.text(
                    "INSERT INTO words_positif (word, value, date, id_category) "
                    "VALUES(:word, :value, :date, :category)"
                )
                conn.execute(
                    stmt,
                    word = tag,
                    value = tags.get(tag),
                    date = date_sql_now,
                    category = category
                )

        for category in cnt_tags_negative:
            tags = cnt_tags_negative.get(category)
            for tag in tags:
                stmt = sqlalchemy.text(
                    "INSERT INTO words_negatif (word, value, date, id_category) "
                    "VALUES(:word, :value, :date, :category)"
                )
                conn.execute(
                    stmt,
                    word = tag,
                    value = tags.get(tag),
                    date = date_sql_now,
                    category = category
                )


if __name__ == '__main__':
    full_predict({'attributes': {'date': '05/05/2021'}}, 'context')
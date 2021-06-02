import os
import tensorflow as tf
import pandas as pd
from google.cloud import storage
from datetime import datetime
from scraper.detik import get_detik_dataframe_from_date
from scraper.kompas import get_kompas_dataframe_from_date

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
def clean_text(text):
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

def vectorize_verif(data):
  bow_data = CountVectorizer.transform(data['content'])
  bow_data = pd.DataFrame.sparse.from_spmatrix(bow_data,columns=CountVectorizer.get_feature_names())
  return bow_data

def predict_data(data):
  data['content'] = data['content'].apply(clean_text)
  bow_data = vectorize_verif(data)
  return model.predict(bow_data).round()

## Get news
date_now = datetime.today().strftime('%d/%m/%Y')
detik_news = get_detik_dataframe_from_date(date_now)
kompas_news = get_kompas_dataframe_from_date(date_now)
news_data = pd.concat([detik_news, kompas_news]).reset_index(drop=True)

## Preprocess
news_data['content'] = detik_news['content'].apply(get_content)
news_data['Len'] = news_data['content'].apply(check_len)
news_data = news_data[news_data['Len'] > 10].reset_index(drop=True)

## Load tensorflow model
dest_folder = '/tmp/'
download_dir = download_model(
    bucket_name = os.environ['BUCKET_NAME'],
    source_dir = 'model',
    destination_folder = dest_folder,
    model_version = '1'
)

dest_vectorizer = os.path.join(dest_folder, 'CountVectorizer.pkl')
download_vectorizer = download_blob(
    bucket_name = os.environ['BUCKET_NAME'],
    source_blob_name = 'CountVectorizer.pkl',
    destination_file_name = dest_folder
)

CountVectorizer = joblib.load(dest_vectorizer)  
stop_words = open("stopword.txt", "r").read().split()
model = tf.keras.models.load_model('model')

pred = predict_data(news_data)
print(pred)

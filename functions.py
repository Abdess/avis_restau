import os
import re

import cv2 as cv
import numpy as np
import pandas as pd
import requests
import spacy
from keras.applications.vgg16 import VGG16
from sklearn import metrics

nlp = spacy.load("fr_core_news_lg")
nlp.add_pipe('sentencizer')

model = VGG16()
PATH = 'data/raw/yelp_photos/photos/'


# from joblib import Parallel, delayed

def appendDFToCSV_void(df, csvFilePath, sep=","):
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(
            pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception(
            "Les colonnes ne correspondent pas ! Le dataframe a " +
            str(len(df.columns)) + " colonnes. Le fichier CSV a " +
            str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) +
            " colonnes.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1,
                                        sep=sep).columns).all():
        raise Exception(
            "Les colonnes et l'ordre des colonnes du dataframe et du fichier csv ne correspondent pas !"
        )
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)


def build_histogram(kmeans, des, image_num):
    res = kmeans.predict(des)
    hist = np.zeros(len(kmeans.cluster_centers_))
    nb_des = len(des)
    if nb_des == 0:
        print("error  : ", image_num)
    for i in res:
        hist[i] += 1.0 / nb_des
    return hist


def createHist(imgName, kmeans):
    # Extract desc
    img = cv.imread(PATH + imgName)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)

    # Build hist
    hist = build_histogram(kmeans, desc, 0)

    return hist


def conf_mat_transform(y_true, y_pred):
    conf_mat = metrics.confusion_matrix(y_true, y_pred)

    # Utilisez la matrice de confusion de la cellule située au-dessus de
    # celle-ci pour décider du corresp. Vous pouvez faire quelque chose
    # comme 'argmax' mais vous ne pouvez pas utiliser deux fois l'ID de
    # la colonne. Si vous utilisez argmax(), cela peut arriver.
    corresp = [3, 0, 4, 2, 1]
    labels = pd.Series(y_true, name="y_true").to_frame()
    labels['y_pred'] = y_pred
    labels['y_pred_transform'] = labels['y_pred'].apply(lambda x: corresp[x])

    return labels['y_pred_transform']


def download(row):
    root_folder = 'data/raw/photos/'

    rm_base_url = re.sub('http.*o/', '', str(row.photos))

    name_cleaned = rm_base_url.replace("/", "_")

    filename = os.path.join(root_folder, name_cleaned)

    # créer un dossier s'il n'existe pas
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    url = row.photos
    print(f"Téléchargement de {url} vers {filename}")
    r = requests.get(url, allow_redirects=True)
    with open(filename, 'wb') as f:
        f.write(r.content)


def selected_topics(model, vectorizer, top_n=10):
    '''Fonctions permettant d'afficher des mots-clés pour chaque topic'''
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names_out()[i], topic[i])
               for i in topic.argsort()[:-top_n - 1:-1]])

###
### NLP Pipeline
###

# def chunker(iterable, total_length, chunksize):
#     return (iterable[pos:pos + chunksize]
#             for pos in range(0, total_length, chunksize))


# def flatten(list_of_lists):
#     "Flatten a list of lists to a combined list"
#     return [item for sublist in list_of_lists for item in sublist]


# def lemmatize_pipe(doc):
#     lemma_list = [
#         str(tok.lemma_).lower() for tok in doc
#         if tok.is_alpha and tok.text.lower() not in stopwords
#     ]
#     return lemma_list


# def process_chunk(texts):
#     preproc_pipe = []
#     for doc in nlp.pipe(texts, batch_size=20):
#         preproc_pipe.append(lemmatize_pipe(doc))
#     return preproc_pipe


# def preprocess_parallel(df_preproc, texts, chunksize=100):
#     executor = Parallel(n_jobs=7,
#                         backend='multiprocessing',
#                         prefer="processes")
#     do = delayed(process_chunk)
#     tasks = (do(chunk)
#              for chunk in chunker(texts, len(df_preproc), chunksize=chunksize))
#     result = executor(tasks)
#     return flatten(result)


# def preprocess_pipe(texts):
#     preproc_pipe = []
#     for doc in nlp.pipe(texts, batch_size=20):
#         preproc_pipe.append(lemmatize_pipe(doc))
#     return preproc_pipe


# def get_stopwords():
#     "Renvoie un ensemble de stopwords lus à partir d'un fichier."
#     with open(stopwordfile) as f:
#         stopwords = []
#         for line in f:
#             stopwords.append(line.strip("\n"))
#     # Convertir en un ensemble pour la performance
#     stopwords_set = set(stopwords)
#     return stopwords_set


# stopwords = get_stopwords()

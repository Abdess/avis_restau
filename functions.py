import os
import pandas as pd
import re
import requests
# import spacy
# from joblib import Parallel, delayed

# nlp = spacy.load('fr_core_news_lg')
# nlp.add_pipe('sentencizer')


def appendDFToCSV_void(df, csvFilePath, sep=","):
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception("Les colonnes ne correspondent pas ! Le dataframe a " + str(len(df.columns)) +
                        " colonnes. Le fichier CSV a " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " colonnes.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception(
            "Les colonnes et l'ordre des colonnes du dataframe et du fichier csv ne correspondent pas !")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)


def selected_topics(model, vectorizer, top_n=10):
    '''Fonctions permettant d'afficher des mots-clés pour chaque topic'''
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names_out()[i], topic[i])
               for i in topic.argsort()[:-top_n - 1:-1]])


def download(row):

    root_folder = 'data/raw/photos/'

    rm_base_url = re.sub('http.*o/', '', str(row.photos))

    name_cleaned = rm_base_url.replace("/", "_")

    filename = os.path.join(root_folder,
                            name_cleaned)

    # créer un dossier s'il n'existe pas
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    url = row.photos
    print(f"Téléchargement de {url} vers {filename}")
    r = requests.get(url, allow_redirects=True)
    with open(filename, 'wb') as f:
        f.write(r.content)

# def chunker(iterable, total_length, chunksize):
#     return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

# def flatten(list_of_lists):
#     "Flatten a list of lists to a combined list"
#     return [item for sublist in list_of_lists for item in sublist]

# def lemmatize_pipe(doc):
#     lemma_list = [str(tok.lemma_).lower() for tok in doc
#                   if tok.is_alpha and tok.text.lower() not in stopwords]
#     return lemma_list

# def process_chunk(texts):
#     preproc_pipe = []
#     for doc in nlp.pipe(texts, batch_size=20):
#         preproc_pipe.append(lemmatize_pipe(doc))
#     return preproc_pipe

# def preprocess_parallel(df_preproc, texts, chunksize=100):
#     executor = Parallel(n_jobs=7, backend='multiprocessing', prefer="processes")
#     do = delayed(process_chunk)
#     tasks = (do(chunk) for chunk in chunker(texts, len(df_preproc), chunksize=chunksize))
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

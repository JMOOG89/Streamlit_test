import streamlit as st
import numpy as np
from sklearn.preprocessing import Normalizer
from keras.models import load_model
import tensorflow as tf
import pandas as pd
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---- Reproduire le prétraitement pour récupérer word2idx, idx2word, vocab_size ----
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
stop_words = stopwords.words('english')

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!]+", " ", w)
    w = re.sub(r'\b\w{0,2}\b', '', w)
    mots = word_tokenize(w.strip())
    mots = [mot for mot in mots if mot not in stop_words]
    return ' '.join(mots).strip()

df = pd.read_csv("MovieReview.csv", sep=",", on_bad_lines='skip')
df = df.drop('sentiment', axis=1)
df.review = df.review.apply(lambda x: preprocess_sentence(x))

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df.review)
word2idx = tokenizer.word_index
idx2word = tokenizer.index_word
vocab_size = tokenizer.num_words

# ---- Charger le modèle ----
model = load_model("word2vec.h5")
vectors = model.layers[0].trainable_weights[0].numpy()

# ---- Fonctions ----
def cosine_similarity(vec1, vec2):
    return np.sum(vec1*vec2) / np.sqrt(np.sum(vec1*vec1) * np.sum(vec2*vec2))

def find_closest(word_index, vectors, number_closest):
    list1 = []
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist, index])
    return np.asarray(sorted(list1, reverse=True)[:number_closest])

# ---- Interface Streamlit ----
st.title("Modèle Word2Vec")
word = st.text_input("Entrez un mot :")
if word and word in word2idx:
    results = find_closest(word2idx[word], vectors, 10)
    for row in results:
        st.write(f"{idx2word[row[1]]} -- {row[0]:.4f}")
elif word:
    st.write("Mot non trouvé dans le vocabulaire.")
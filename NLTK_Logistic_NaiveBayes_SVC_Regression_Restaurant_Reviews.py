#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !jupyter nbconvert --to script NLTK_Logistic_NaiveBayes_SVC_Regression_Restaurant_Reviews.ipynb


# In[2]:


# pip install tensorflow


# In[3]:


import re
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import nltk


# FOR LSTM
# try:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
# except Exception:
#     tf = None
#     Tokenizer = None
#     pad_sequences = None


# In[4]:


# nltk.download('punkt_tab')    # for tokenization
# nltk.download('stopwords')    # for stopwords removal


# ---- ensure NLTK data and then create stopwords/stemmer ----
nltk.download('punkt', quiet=True)       # tokenizers/punkt
nltk.download('stopwords', quiet=True)   # corpora/stopwords


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()



# In[5]:


# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()


# ## **Clean and normalize text for ML.**
# Makes text more uniform, reduces noise, and simplifies vocabulary for model training.

# In[6]:


def preprocess(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = text.split()

    negation_words = {"not", "no", "nor", "n't"}
    tokens = [word for word in tokens if word not in stop_words or word in negation_words]

    # tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    return ' '.join(tokens)


# In[7]:


# from google.colab import drive
# drive.mount('/content/drive')



# In[8]:


import csv

csv_path = './/Restaurant_Reviews.tsv'

# .tsv to csv
# with open(csv_path, 'r', newline='') as tsvfile, open('/content/drive/My Drive/NLP/Restaurant_Reviews.csv', 'w', newline='') as csvfile:
#     tsv_reader = csv.reader(tsvfile, delimiter='\t')
#     csv_writer = csv.writer(csvfile, delimiter=',')
#     for row in tsv_reader:
#         csv_writer.writerow(row)


# In[9]:


# df = pd.DataFrame(data)
df = pd.read_csv("./Restaurant_Reviews.tsv", sep='\t')
df


# In[10]:


df['clean_Review'] = df['Review'].apply(preprocess)
df


# # **TF-IDF vectorisation**
# converts text into numerical vectors

# In[11]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_Review'])
y = df['Liked']

# map integer class indices back to original labels (useful for multiclass)
label_classes = np.unique(y)


# In[12]:


y


# In[13]:


X


# # **Model Training**

# In[14]:


# ---- BiLSTM: tokenizer + sequence prep (replace existing tokenizer block) ----
vocab_size = 20000
maxlen = 128
oov_token = "<OOV>"

# fallback: if Tokenizer not available from tf.keras, try keras package
if Tokenizer is None:
    try:
        from keras.preprocessing.text import Tokenizer as TokenizerLocal
        from keras.preprocessing.sequence import pad_sequences as pad_sequences_local
        Tokenizer = TokenizerLocal
        pad_sequences = pad_sequences_local
    except Exception:
        Tokenizer = None
        pad_sequences = None

if Tokenizer is not None:
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    # Use the cleaned review column you created earlier
    if "clean_Review" in df.columns:
        texts_for_tokenizer = df["clean_Review"].astype(str).tolist()
    else:
        # fallback: apply preprocess to the original review column
        texts_for_tokenizer = df["Review"].astype(str).apply(preprocess).tolist()

    tokenizer.fit_on_texts(texts_for_tokenizer)
    sequences = tokenizer.texts_to_sequences(texts_for_tokenizer)
    X_seq_all = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
else:
    tokenizer = None
    X_seq_all = None


# In[15]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# keep existing TF-IDF split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y))>1 else None)

# split sequences for BiLSTM (if available)
if X_seq_all is not None:
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_seq_all, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y))>1 else None)
else:
    Xs_train = Xs_test = ys_train = ys_test = None


# In[16]:


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n====== {name} ======")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model


# In[17]:


lr_model = train_and_evaluate_model(LogisticRegression(), X_train, y_train, X_test, y_test, name="Logistic Regression")
nb_model = train_and_evaluate_model(MultinomialNB(), X_train, y_train, X_test, y_test, name="Naive Bayes")
svc_model = train_and_evaluate_model(SVC(kernel='linear'), X_train, y_train, X_test, y_test, name="Support Vector Classifier")


# ---- Train BiLSTM ----
bilstm_model = None
if tf is None:
    print("TensorFlow not available — skipping BiLSTM.")
else:
    if Xs_train is None:
        print("No tokenized sequence data (Xs_train is None). Ensure tokenizer and X_seq_all were created.")
    else:
        embed_dim = 100
        lstm_units = 128
        batch_size = 64
        epochs = 6

        n_classes = len(np.unique(ys_train))
        if n_classes > 2:
            # multiclass
            bilstm_model = Sequential([
                Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen),
                Bidirectional(LSTM(lstm_units, return_sequences=False)),
                Dropout(0.4),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(n_classes, activation='softmax')
            ])
            bilstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            # binary
            bilstm_model = Sequential([
                Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen),
                Bidirectional(LSTM(lstm_units, return_sequences=False)),
                Dropout(0.4),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            bilstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = bilstm_model.fit(
            Xs_train, ys_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
            verbose=1
        )

        # Evaluation
        probs = bilstm_model.predict(Xs_test, verbose=0)
        probs = np.array(probs)
        if probs.ndim == 1:
            probs = probs.reshape((-1,1))
        if probs.shape[1] > 1:
            y_pred_bilstm = probs.argmax(axis=1)
        else:
            y_pred_bilstm = (probs[:,0] >= 0.5).astype(int)

        print("\n====== BiLSTM ======")
        print("Accuracy:", accuracy_score(ys_test, y_pred_bilstm))
        print("Classification Report:\n", classification_report(ys_test, y_pred_bilstm))


# In[18]:


# def predict_sentiment(text, model):
#     clean = preprocess(text)
#     vec = vectorizer.transform([clean])
#     result = model.predict(vec)[0]
#     return "Positive" if result == 1 else "Negative"


def predict_sentiment(text, model, vectorizer=vectorizer, preprocess_flag=True):
    txt = preprocess(text) if preprocess_flag else str(text)
    vec = vectorizer.transform([txt])
    pred = model.predict(vec)[0]
    # If label_classes has more than two items, return mapped label
    try:
        if len(label_classes) > 2:
            return label_classes[pred]
    except Exception:
        # if label_classes or mapping fails, fall back to numeric return
        pass
    # binary mapping (assumes positive label is 1)
    return "Positive" if int(pred) == 1 else "Negative"


def predict_sentiment_bilstm(text, model=bilstm_model, tokenizer=tokenizer, maxlen=maxlen, preprocess_flag=True):
    if model is None or tokenizer is None:
        return "(bilstm not available)"
    txt = preprocess(text) if preprocess_flag else str(text)
    seq = tokenizer.texts_to_sequences([txt])
    pad = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    probs = model.predict(pad, verbose=0)
    probs = np.array(probs)
    if probs.ndim == 1:
        probs = probs.reshape((-1,1))
    if probs.shape[1] > 1:
        pred = int(probs.argmax(axis=1)[0])
        # map predicted class index back to original label if label_classes exists
        try:
            return label_classes[pred]
        except Exception:
            return int(pred)
  # multiclass: returns class index (int). Map to labels if you have label names.
    else:
        pred = 1 if probs[0,0] >= 0.5 else 0
        return "Positive" if pred == 1 else "Negative"


# In[19]:


test_input = "I am not happy with the service"

for model, name in zip([lr_model, nb_model, svc_model], ["LogisticRegression", "NaiveBayes", "SVC"]):
    print(f"\n{name} Prediction:")
    print(f"Input: '{test_input}'")
    print("Predicted Sentiment:", predict_sentiment(test_input, model))


# Save tokenizer and bilstm (if you want persistence)
import joblib, os
out_dir = "model_artifacts"
os.makedirs(out_dir, exist_ok=True)
if tokenizer is not None:
    try:
        joblib.dump(tokenizer, f"{out_dir}/tokenizer.joblib")
    except Exception:
        print("Failed to save tokenizer via joblib.")
if bilstm_model is not None and tf is not None:
    try:
        bilstm_model.save(os.path.join(out_dir, "bilstm_model"), overwrite=True, include_optimizer=False)
    except Exception:
        bilstm_model.save(os.path.join(out_dir, "bilstm_model.h5"), overwrite=True)


# In[ ]:





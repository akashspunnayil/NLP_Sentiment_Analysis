# sentiment_analysis_app.py

# --- Force NLTK to use bundled nltk_data folder (if present) ---
import os
HERE = os.path.dirname(__file__)
os.environ.setdefault("NLTK_DATA", os.path.join(HERE, "nltk_data"))

# --- Regular imports ---
import streamlit as st
import pandas as pd
import re
import string
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# remaining imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- Keras / TF for BiLSTM (add near other imports) ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping
except Exception as e:
    tf = None
    # We'll check tf later before training/predicting

st.set_page_config(page_title="Sentiment Analysis (TF-IDF + ML)", layout="wide")
st.title("Sentiment Analysis — TF-IDF + ML (LogReg, NB, SVC, BiLSTM)")

# ---- ensure NLTK resources ----
with st.spinner("Checking NLTK data..."):
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
negation_words = {"not", "no", "nor", "n't"}

# ---- preprocessing ----
def preprocess(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    # keep negations even if they are stopwords
    tokens = [w for w in tokens if (w not in stop_words) or (w in negation_words)]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# ---- Sidebar: upload + params ----
st.sidebar.header("Data & Settings")
uploaded = st.sidebar.file_uploader("Upload CSV or TSV", type=["csv", "tsv", "txt"])
use_sample = st.sidebar.checkbox("Use built-in sample", value=False)

sep_label = st.sidebar.selectbox(
    "Separator",
    options=["Comma (,)", "Tab (\\t)", "Semicolon (;)"],
    index=0
)

if sep_label.startswith("Comma"):
    sep_opt = ","
elif sep_label.startswith("Tab"):
    sep_opt = "\t"
else:
    sep_opt = ";"

apply_preprocess = st.sidebar.checkbox(
    "Apply preprocessing to text",
    value=True,
    help="If unchecked, raw text will be used (no lowercasing/stopword/stemming)."
)

st.sidebar.markdown("---")
st.sidebar.markdown("Model / split settings")
test_size = st.sidebar.slider("Test set fraction", 0.1, 0.5, 0.3, 0.05)
random_state = int(st.sidebar.number_input("Random state", value=42, step=1))

st.sidebar.markdown("---")
st.sidebar.markdown("Save / Load artifacts")
artifact_dir_default = "model_artifacts"
artifact_dir_input = st.sidebar.text_input("Artifacts directory (relative)", value=artifact_dir_default)

# ---- Load data ----
if use_sample or uploaded is None:
    sample = pd.DataFrame({
        "Review": [
            "I love this place",
            "Worst food ever",
            "Not bad could be better",
            "Absolutely fantastic experience",
            "Terrible service and boring food",
            "I really enjoyed it",
            "Food was cold when served",
            "Highly unhygienic staff",
            "Average experience",
            "Satisfying meal"
        ],
        "Liked": [1, 0, 1, 1, 0, 1, 0, 0, 2, 2]  # example with a 3-class label (0,1,2)
    })
    df = sample
    st.sidebar.info("Using built-in sample. Upload a file to use your dataset.")
else:
    try:
        df = pd.read_csv(uploaded, sep=sep_opt)
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")
        st.stop()

st.subheader("Data preview")
st.write(f"Rows: {len(df)}")
st.dataframe(df.head(10))

cols = df.columns.tolist()
if len(cols) == 0:
    st.error("No columns detected in the uploaded file.")
    st.stop()

text_col = st.sidebar.selectbox("Text column", options=cols, index=0)
label_col = st.sidebar.selectbox("Label column", options=cols, index=min(1, len(cols)-1))

unique_labels = sorted(df[label_col].dropna().unique().tolist())
st.sidebar.markdown("Select labels to include (from data)")
if len(unique_labels) == 0:
    st.sidebar.warning("No labels found in the selected label column.")
    selected_labels = []
else:
    selected_labels = st.sidebar.multiselect("Pick labels", options=unique_labels, default=unique_labels)

# ---- Prepare data ----
if st.button("Preprocess & Split"):
    if selected_labels:
        df_work = df[df[label_col].isin(selected_labels)].copy()
        if df_work.empty:
            st.error("No rows match the selected labels. Pick different labels or check your data.")
            st.stop()
    else:
        df_work = df.copy()

    if apply_preprocess:
        df_work["_clean_text_"] = df_work[text_col].astype(str).apply(preprocess)
    else:
        df_work["_clean_text_"] = df_work[text_col].astype(str)

    n_before = len(df_work)
    df_work = df_work.dropna(subset=[label_col])
    if len(df_work) < n_before:
        st.warning(f"Dropped {n_before-len(df_work)} rows with missing labels.")
    labels_series = df_work[label_col]

    st.write("Label distribution (after filtering):")
    st.write(labels_series.value_counts())

    # --- Vectorize for TF-IDF (existing) ---
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_work["_clean_text_"])

    # --- Also prepare tokenizer + sequences for BiLSTM (new) ---
    vocab_size = 20000
    maxlen = 128
    oov_token = "<OOV>"

    # Tokenizer class exists only if TF imported successfully; handle gracefully
    if tf is None:
        # Attempt to import Tokenizer from keras.preprocessing.text if TF not available
        try:
            from keras.preprocessing.text import Tokenizer as KerasTokenizer
            from keras.preprocessing.sequence import pad_sequences as keras_pad_sequences
            TokenizerLocal = KerasTokenizer
            pad_sequences_local = keras_pad_sequences
        except Exception:
            TokenizerLocal = None
            pad_sequences_local = None
    else:
        TokenizerLocal = Tokenizer
        pad_sequences_local = pad_sequences

    if TokenizerLocal is None:
        tokenizer = None
        X_seq = None
    else:
        tokenizer = TokenizerLocal(num_words=vocab_size, oov_token=oov_token)
        tokenizer.fit_on_texts(df_work["_clean_text_"])
        sequences = tokenizer.texts_to_sequences(df_work["_clean_text_"])
        X_seq = pad_sequences_local(sequences, maxlen=maxlen, padding='post', truncating='post')

    y = labels_series.values

    # split for TF-IDF
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(test_size),
            random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )
    except ValueError as e:
        st.error(f"Error during train-test split: {e}")
        st.stop()

    # split for sequences (BiLSTM)
    if X_seq is not None:
        try:
            Xs_train, Xs_test, ys_train, ys_test = train_test_split(
                X_seq,
                y,
                test_size=float(test_size),
                random_state=random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )
        except ValueError as e:
            st.error(f"Error during sequence train-test split: {e}")
            st.stop()
    else:
        Xs_train = Xs_test = ys_train = ys_test = None

    # store artifacts
    st.session_state["vectorizer"] = vectorizer
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test

    st.session_state["tokenizer"] = tokenizer
    st.session_state["Xs_train"] = Xs_train
    st.session_state["Xs_test"] = Xs_test
    st.session_state["ys_train"] = ys_train
    st.session_state["ys_test"] = ys_test
    st.session_state["seq_params"] = {"vocab_size": vocab_size, "maxlen": maxlen, "oov_token": oov_token}

    st.session_state["apply_preprocess"] = bool(apply_preprocess)
    st.session_state["selected_labels"] = selected_labels
    st.success("Preprocessing and split completed. Ready to train models.")

# ---- Train models ----
def train_and_eval(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    return model, acc, report

st.subheader("Train models")
if st.button("Train LogisticRegression, NaiveBayes, SVC, BiLSTM"):
    if not all(k in st.session_state for k in ("vectorizer", "X_train", "X_test", "y_train", "y_test")):
        st.warning("Run 'Preprocess & Split' first.")
    else:
        X_train = st.session_state["X_train"]
        X_test = st.session_state["X_test"]
        y_train = st.session_state["y_train"]
        y_test = st.session_state["y_test"]

        with st.spinner("Training LogisticRegression..."):
            lr = LogisticRegression(max_iter=2000)
            lr, lr_acc, lr_report = train_and_eval(lr, X_train, y_train, X_test, y_test)
            st.write("LogisticRegression — Accuracy:", lr_acc)
            st.text(lr_report)
            st.session_state["lr_model"] = lr

        with st.spinner("Training MultinomialNB..."):
            nb = MultinomialNB()
            nb, nb_acc, nb_report = train_and_eval(nb, X_train, y_train, X_test, y_test)
            st.write("MultinomialNB — Accuracy:", nb_acc)
            st.text(nb_report)
            st.session_state["nb_model"] = nb

        with st.spinner("Training SVC (linear)..."):
            svc = SVC(kernel="linear", probability=True)
            svc, svc_acc, svc_report = train_and_eval(svc, X_train, y_train, X_test, y_test)
            st.write("SVC — Accuracy:", svc_acc)
            st.text(svc_report)
            st.session_state["svc_model"] = svc

        # BiLSTM training
        with st.spinner("Training Bidirectional LSTM (BiLSTM)..."):
            if tf is None:
                st.warning("TensorFlow not available in this environment — skipping BiLSTM training.")
            else:
                try:
                    Xs_train = st.session_state.get("Xs_train")
                    Xs_test = st.session_state.get("Xs_test")
                    ys_train = st.session_state.get("ys_train")
                    ys_test = st.session_state.get("ys_test")
                    seq_params = st.session_state.get("seq_params", {})
                    vocab_size = seq_params.get("vocab_size", 20000)
                    maxlen = seq_params.get("maxlen", 128)

                    if Xs_train is None or ys_train is None:
                        st.warning("No sequence training data available. Run 'Preprocess & Split' with Tokenizer available.")
                    else:
                        embed_dim = 100
                        lstm_units = 128
                        batch_size = 64
                        epochs = 6

                        n_classes = len(np.unique(ys_train))
                        if n_classes > 2:
                            # multiclass setup
                            bilstm = Sequential([
                                Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen),
                                Bidirectional(LSTM(lstm_units, return_sequences=False)),
                                Dropout(0.4),
                                Dense(64, activation='relu'),
                                Dropout(0.2),
                                Dense(n_classes, activation='softmax')
                            ])
                            bilstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                            history = bilstm.fit(Xs_train, ys_train, validation_split=0.1, epochs=epochs,
                                                 batch_size=batch_size, callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)], verbose=0)
                            loss, acc = bilstm.evaluate(Xs_test, ys_test, verbose=0)
                            # st.write(f"BiLSTM (multiclass) — Accuracy: {acc:.4f}")
                            st.write("BiLSTM (multiclass) — Accuracy:", acc)
                        else:
                            # binary setup
                            bilstm = Sequential([
                                Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen),
                                Bidirectional(LSTM(lstm_units, return_sequences=False)),
                                Dropout(0.4),
                                Dense(64, activation='relu'),
                                Dropout(0.2),
                                Dense(1, activation='sigmoid')
                            ])
                            bilstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                            history = bilstm.fit(Xs_train, ys_train, validation_split=0.1, epochs=epochs,
                                                 batch_size=batch_size, callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)], verbose=0)
                            loss, acc = bilstm.evaluate(Xs_test, ys_test, verbose=0)
                            # st.write(f"BiLSTM (binary) — Accuracy: {acc:.4f}")
                            st.write("BiLSTM (binary) — Accuracy:", acc)

                        st.session_state["bilstm_model"] = bilstm
                except KeyError:
                    st.error("Sequence training data not found in session. Run 'Preprocess & Split' first.")
                except Exception as e:
                    st.error(f"BiLSTM training failed: {e}")

        st.success("Training finished and models saved to session.")

# ---- Line-by-line predictions ----
st.subheader("Predict line-by-line sentences")
st.markdown("Enter one sentence per line. Predictions will be shown for all trained models.")
input_text = st.text_area("Sentences (one per line)", value="The food was great\nService was terrible\nNot happy with the price")

current_pre_flag = st.session_state.get("apply_preprocess", apply_preprocess)
st.caption(f"Using preprocessing: {current_pre_flag}")

if st.button("Predict Sentences"):
    sentences = [s.strip() for s in input_text.split("\n") if s.strip()]
    if len(sentences) == 0:
        st.warning("Enter at least one non-empty sentence.")
    else:
        if "vectorizer" not in st.session_state:
            st.warning("You must run 'Preprocess & Split' (to fit vectorizer) and train models before prediction.")
        else:
            vectorizer = st.session_state["vectorizer"]
            use_pre = st.session_state.get("apply_preprocess", apply_preprocess)

            if use_pre:
                X_new = vectorizer.transform([preprocess(s) for s in sentences])
            else:
                X_new = vectorizer.transform([str(s) for s in sentences])

            results = {"Sentence": sentences}

            # helper to add per-class probs and preds (classical models)
            def add_model_outputs(name, model):
                if model is None:
                    results[f"{name}_pred"] = ["(not trained)"] * len(sentences)
                    return
                preds = model.predict(X_new)
                results[f"{name}_pred"] = preds.tolist()
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_new)
                    classes = model.classes_
                    for i, cls in enumerate(classes):
                        col = f"{name}_prob_{cls}"
                        results[col] = [float(round(v, 4)) for v in probs[:, i].tolist()]
                else:
                    try:
                        dfun = model.decision_function(X_new)
                        if dfun.ndim == 1:
                            probs = 1 / (1 + np.exp(-dfun))
                            results[f"{name}_prob_1"] = [float(round(v, 4)) for v in probs.tolist()]
                        else:
                            exp = np.exp(dfun - np.max(dfun, axis=1, keepdims=True))
                            sm = exp / np.sum(exp, axis=1, keepdims=True)
                            classes = model.classes_
                            for i, cls in enumerate(classes):
                                col = f"{name}_prob_{cls}"
                                results[col] = [float(round(v, 4)) for v in sm[:, i].tolist()]
                    except Exception:
                        pass

            # robust BiLSTM predictions
            def add_bilstm_outputs(name, model, tokenizer, maxlen=128):
                if model is None or tokenizer is None:
                    results[f"{name}_pred"] = ["(not trained)"] * len(sentences)
                    return
                texts = [preprocess(s) for s in sentences] if use_pre else [str(s) for s in sentences]
                seqs = tokenizer.texts_to_sequences(texts)
                padded = pad_sequences(seqs, maxlen=maxlen, padding='post', truncating='post')
                try:
                    probs = model.predict(padded)
                except Exception:
                    results[f"{name}_pred"] = ["(error)"] * len(sentences)
                    return

                probs = np.array(probs)
                if probs.ndim == 1:
                    probs = probs.reshape((-1, 1))

                if probs.shape[1] > 1:
                    preds = probs.argmax(axis=1).tolist()
                    results[f"{name}_pred"] = preds
                    for cls_i in range(probs.shape[1]):
                        results[f"{name}_prob_{cls_i}"] = [float(round(x, 4)) for x in probs[:, cls_i].tolist()]
                else:
                    probs1 = probs[:, 0].tolist()
                    preds = [1 if p >= 0.5 else 0 for p in probs1]
                    results[f"{name}_pred"] = preds
                    results[f"{name}_prob_1"] = [float(round(p, 4)) for p in probs1]

            # call classical model outputs
            add_model_outputs("LogReg", st.session_state.get("lr_model"))
            add_model_outputs("NB", st.session_state.get("nb_model"))
            add_model_outputs("SVC", st.session_state.get("svc_model"))

            # call BiLSTM outputs
            add_bilstm_outputs(
                "BiLSTM",
                st.session_state.get("bilstm_model"),
                st.session_state.get("tokenizer"),
                maxlen=st.session_state.get("seq_params", {}).get("maxlen", 128)
            )

            res_df = pd.DataFrame(results)
            st.dataframe(res_df)

# ---- Save / Load model (optional) ----
st.subheader("Save / Load")

# Read artifacts directory from sidebar session value (no duplicate widget)
out_dir_input = st.session_state.get("artifact_dir_input", artifact_dir_input or artifact_dir_default)

import io, tempfile, shutil

st.write("Files saved to (container):", os.path.abspath(out_dir_input))
if os.path.exists(out_dir_input):
    try:
        st.write("Directory contents:", os.listdir(out_dir_input))
    except Exception:
        st.write("Directory exists but could not list contents.")
else:
    st.write("Directory does not exist yet (will be created on Save).")

col1, col2 = st.columns(2)

with col1:
    if st.button("Save models & vectorizer to disk (pickle)", key="save_disk_btn"):
        if "vectorizer" not in st.session_state:
            st.warning("Nothing to save. Fit vectorizer and train models first.")
        else:
            out_dir = out_dir_input or artifact_dir_default
            os.makedirs(out_dir, exist_ok=True)

            joblib.dump(st.session_state["vectorizer"], f"{out_dir}/vectorizer.joblib")
            for name in ("lr_model", "nb_model", "svc_model"):
                if name in st.session_state:
                    joblib.dump(st.session_state[name], f"{out_dir}/{name}.joblib")

            if "tokenizer" in st.session_state:
                try:
                    joblib.dump(st.session_state["tokenizer"], f"{out_dir}/tokenizer.joblib")
                except Exception:
                    st.warning("Failed to save tokenizer via joblib.")

            if "bilstm_model" in st.session_state:
                bilstm_path = os.path.join(out_dir, "bilstm_model")
                try:
                    st.session_state["bilstm_model"].save(bilstm_path, overwrite=True, include_optimizer=False)
                except Exception:
                    try:
                        st.session_state["bilstm_model"].save(f"{out_dir}/bilstm_model.h5", overwrite=True)
                    except Exception as e:
                        st.warning(f"Failed to save BiLSTM model: {e}")

            meta = {
                "apply_preprocess": bool(st.session_state.get("apply_preprocess", apply_preprocess)),
                "selected_labels": st.session_state.get("selected_labels", selected_labels),
                "seq_params": st.session_state.get("seq_params", {})
            }
            joblib.dump(meta, f"{out_dir}/meta.joblib")
            st.success(f"Saved artifacts to {out_dir}")

    st.markdown("**Download trained artifacts**")
    if "vectorizer" in st.session_state:
        buf = io.BytesIO()
        joblib.dump(st.session_state["vectorizer"], buf)
        st.download_button(
            label="Download Vectorizer",
            data=buf.getvalue(),
            file_name="vectorizer.joblib",
            mime="application/octet-stream",
            key="dl_vectorizer"
        )

    for name in ("lr_model", "nb_model", "svc_model"):
        if name in st.session_state:
            buf = io.BytesIO()
            joblib.dump(st.session_state[name], buf)
            st.download_button(
                label=f"Download {name}",
                data=buf.getvalue(),
                file_name=f"{name}.joblib",
                mime="application/octet-stream",
                key=f"dl_{name}"
            )

    if "tokenizer" in st.session_state:
        buf = io.BytesIO()
        joblib.dump(st.session_state["tokenizer"], buf)
        st.download_button(
            label="Download Tokenizer",
            data=buf.getvalue(),
            file_name="tokenizer.joblib",
            mime="application/octet-stream",
            key="dl_tokenizer"
        )

    if "bilstm_model" in st.session_state:
        out_dir = out_dir_input or artifact_dir_default
        bilstm_folder = os.path.join(out_dir, "bilstm_model")
        bilstm_h5 = os.path.join(out_dir, "bilstm_model.h5")
        if os.path.exists(bilstm_folder):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            tmp.close()
            try:
                shutil.make_archive(tmp.name.replace(".zip", ""), 'zip', bilstm_folder)
                with open(tmp.name, "rb") as f:
                    data = f.read()
                st.download_button(
                    label="Download BiLSTM (SavedModel .zip)",
                    data=data,
                    file_name="bilstm_model_savedmodel.zip",
                    mime="application/zip",
                    key="dl_bilstm_savedmodel"
                )
            finally:
                try:
                    os.remove(tmp.name)
                except Exception:
                    pass
        elif os.path.exists(bilstm_h5):
            with open(bilstm_h5, "rb") as f:
                data = f.read()
            st.download_button(
                label="Download BiLSTM (.h5)",
                data=data,
                file_name="bilstm_model.h5",
                mime="application/octet-stream",
                key="dl_bilstm_h5"
            )

with col2:
    if st.button("Load models & vectorizer from disk (pickle)", key="load_disk_btn"):
        in_dir = out_dir_input or artifact_dir_default
        try:
            st.session_state["vectorizer"] = joblib.load(f"{in_dir}/vectorizer.joblib")
            for name in ("lr_model", "nb_model", "svc_model"):
                path = f"{in_dir}/{name}.joblib"
                try:
                    st.session_state[name] = joblib.load(path)
                except Exception:
                    st.warning(f"Could not load {path}")

            try:
                meta = joblib.load(f"{in_dir}/meta.joblib")
                st.session_state["apply_preprocess"] = bool(meta.get("apply_preprocess", apply_preprocess))
                st.session_state["selected_labels"] = meta.get("selected_labels", selected_labels)
                st.session_state["seq_params"] = meta.get("seq_params", {})
            except Exception:
                pass

            try:
                st.session_state["tokenizer"] = joblib.load(f"{in_dir}/tokenizer.joblib")
            except Exception:
                pass

            try:
                import tensorflow as _tf
                bpath = f"{in_dir}/bilstm_model"
                if os.path.exists(bpath):
                    st.session_state["bilstm_model"] = _tf.keras.models.load_model(bpath)
                elif os.path.exists(f"{in_dir}/bilstm_model.h5"):
                    st.session_state["bilstm_model"] = _tf.keras.models.load_model(f"{in_dir}/bilstm_model.h5")
                else:
                    pass
            except Exception:
                st.warning("Could not load BiLSTM model (TensorFlow may not be available or model files corrupted).")

            st.success("Loaded artifacts (if present).")
        except FileNotFoundError:
            st.error("Directory or files not found.")

# ---- Notes ----
st.markdown("---")
st.markdown(
    """
- You can select which labels from your dataset to include for training using the label multiselect in the sidebar.
- The app supports multi-class labels (strings or integers). Per-class probabilities are shown when available.
- If preprocessing is disabled, TF-IDF is fitted on raw text strings; predictions should use the same mode.
- Save/Load persists the vectorizer, models and metadata (`apply_preprocess` and `selected_labels`).
"""
)

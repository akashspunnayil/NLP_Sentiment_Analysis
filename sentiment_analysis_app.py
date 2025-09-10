# sentiment_analysis_app.py
import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib
import os

st.set_page_config(page_title="Sentiment Analysis (TF-IDF + ML)", layout="wide")
st.title("Sentiment Analysis — TF-IDF + ML (LogReg, NB, SVC)")

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

# friendly separator labels visible to user
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

# New: option to apply preprocessing or not
apply_preprocess = st.sidebar.checkbox(
    "Apply preprocessing to text",
    value=True,
    help="If unchecked, raw text will be used (no lowercasing/stopword/stemming)."
)

st.sidebar.markdown("---")
st.sidebar.markdown("Model / split settings")
test_size = st.sidebar.slider("Test set fraction", 0.1, 0.5, 0.3, 0.05)
random_state = int(st.sidebar.number_input("Random state", value=42, step=1))

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

# column selectors
cols = df.columns.tolist()
if len(cols) == 0:
    st.error("No columns detected in the uploaded file.")
    st.stop()

text_col = st.sidebar.selectbox("Text column", options=cols, index=0)
label_col = st.sidebar.selectbox("Label column", options=cols, index=min(1, len(cols)-1))

# Show unique labels and let user choose which labels to include
unique_labels = sorted(df[label_col].dropna().unique().tolist())
st.sidebar.markdown("Select labels to include (from data)")
if len(unique_labels) == 0:
    st.sidebar.warning("No labels found in the selected label column.")
    selected_labels = []
else:
    # default: select all
    selected_labels = st.sidebar.multiselect("Pick labels", options=unique_labels, default=unique_labels)

st.sidebar.markdown("---")
st.sidebar.markdown("Save / Load artifacts")
artifact_dir_default = "model_artifacts"
artifact_dir_input = st.sidebar.text_input("Artifacts directory (relative)", value=artifact_dir_default)

# ---- Prepare data ----
if st.button("Preprocess & Split"):
    # filter to selected labels if any chosen
    if selected_labels:
        df_work = df[df[label_col].isin(selected_labels)].copy()
        if df_work.empty:
            st.error("No rows match the selected labels. Pick different labels or check your data.")
            st.stop()
    else:
        df_work = df.copy()  # no filtering if user didn't pick labels

    # apply or skip preprocessing
    if apply_preprocess:
        df_work["_clean_text_"] = df_work[text_col].astype(str).apply(preprocess)
    else:
        df_work["_clean_text_"] = df_work[text_col].astype(str)

    # drop rows with missing labels
    n_before = len(df_work)
    df_work = df_work.dropna(subset=[label_col])
    if len(df_work) < n_before:
        st.warning(f"Dropped {n_before-len(df_work)} rows with missing labels.")
    # keep labels as-is (string or numeric) for multi-class
    labels_series = df_work[label_col]

    st.write("Label distribution (after filtering):")
    st.write(labels_series.value_counts())

    # Vectorize
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_work["_clean_text_"])
    y = labels_series.values

    # split
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

    # store artifacts in session
    st.session_state["vectorizer"] = vectorizer
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
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
if st.button("Train LogisticRegression, NaiveBayes, SVC"):
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

        st.success("Training finished and models saved to session.")

# ---- Line-by-line predictions ----
st.subheader("Predict line-by-line sentences")
st.markdown("Enter one sentence per line. Predictions will be shown for all trained models.")
input_text = st.text_area("Sentences (one per line)", value="The food was great\nService was terrible\nNot happy with the price")

# show current preprocess mode (from session if set, else sidebar value)
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

            # helper to add per-class probs and preds
            def add_model_outputs(name, model):
                if model is None:
                    results[f"{name}_pred"] = ["(not trained)"] * len(sentences)
                    return
                # predictions
                preds = model.predict(X_new)
                results[f"{name}_pred"] = preds.tolist()
                # probabilities if available -> multi-class support
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_new)  # shape n_samples x n_classes
                    classes = model.classes_
                    for i, cls in enumerate(classes):
                        col = f"{name}_prob_{cls}"
                        results[col] = [float(round(v, 4)) for v in probs[:, i].tolist()]
                else:
                    # fallback: decision_function -> convert to pseudo-prob using sigmoid for binary,
                    # or softmax-like scaling for multi-class
                    try:
                        dfun = model.decision_function(X_new)
                        if dfun.ndim == 1:
                            # binary: sigmoid
                            probs = 1 / (1 + np.exp(-dfun))
                            results[f"{name}_prob_1"] = [float(round(v, 4)) for v in probs.tolist()]
                        else:
                            # multi-class: softmax
                            exp = np.exp(dfun - np.max(dfun, axis=1, keepdims=True))
                            sm = exp / np.sum(exp, axis=1, keepdims=True)
                            classes = model.classes_
                            for i, cls in enumerate(classes):
                                col = f"{name}_prob_{cls}"
                                results[col] = [float(round(v, 4)) for v in sm[:, i].tolist()]
                    except Exception:
                        # no probabilities available
                        pass

            add_model_outputs("LogReg", st.session_state.get("lr_model"))
            add_model_outputs("NB", st.session_state.get("nb_model"))
            add_model_outputs("SVC", st.session_state.get("svc_model"))

            res_df = pd.DataFrame(results)
            st.dataframe(res_df)

# ---- Save / Load model (optional) ----
st.subheader("Save / Load")
out_dir_input = st.text_input("Artifacts directory (relative)", value=artifact_dir_input or artifact_dir_default)

col1, col2 = st.columns(2)
with col1:
    if st.button("Save models & vectorizer to disk (pickle)"):
        if "vectorizer" not in st.session_state:
            st.warning("Nothing to save. Fit vectorizer and train models first.")
        else:
            out_dir = out_dir_input or artifact_dir_default
            os.makedirs(out_dir, exist_ok=True)
            joblib.dump(st.session_state["vectorizer"], f"{out_dir}/vectorizer.joblib")
            for name in ("lr_model", "nb_model", "svc_model"):
                if name in st.session_state:
                    joblib.dump(st.session_state[name], f"{out_dir}/{name}.joblib")
            meta = {
                "apply_preprocess": bool(st.session_state.get("apply_preprocess", apply_preprocess)),
                "selected_labels": st.session_state.get("selected_labels", selected_labels),
            }
            joblib.dump(meta, f"{out_dir}/meta.joblib")
            st.success(f"Saved artifacts to {out_dir}")

with col2:
    if st.button("Load models & vectorizer from disk (pickle)"):
        in_dir = out_dir_input or artifact_dir_default
        try:
            st.session_state["vectorizer"] = joblib.load(f"{in_dir}/vectorizer.joblib")
            for name in ("lr_model", "nb_model", "svc_model"):
                path = f"{in_dir}/{name}.joblib"
                try:
                    st.session_state[name] = joblib.load(path)
                except Exception:
                    st.warning(f"Could not load {path}")
            # load metadata if present
            try:
                meta = joblib.load(f"{in_dir}/meta.joblib")
                st.session_state["apply_preprocess"] = bool(meta.get("apply_preprocess", apply_preprocess))
                st.session_state["selected_labels"] = meta.get("selected_labels", selected_labels)
            except Exception:
                pass
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


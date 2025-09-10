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

sep_opt = st.sidebar.selectbox("Separator", options=[",", "\t"], index=0)

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
            "Highly unhygienic staff"
        ],
        "Liked": [1, 0, 1, 1, 0, 1, 0, 0]
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
text_col = st.sidebar.selectbox("Text column", options=cols, index=0)
label_col = st.sidebar.selectbox("Label column (binary 0/1)", options=cols, index=min(1, len(cols)-1))

# model params
test_size = st.sidebar.slider("Test set fraction", 0.1, 0.5, 0.3, 0.05)
random_state = int(st.sidebar.number_input("Random state", value=42, step=1))

# ---- Prepare data ----
if st.button("Preprocess & Split"):
    df["_clean_text_"] = df[text_col].astype(str).apply(preprocess)
    # coerce labels to integers 0/1
    try:
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    except Exception:
        df[label_col] = df[label_col].astype(str).map(lambda x: 1 if str(x).lower() in ("1","yes","y","true","positive","pos") else 0)
    n_before = len(df)
    df = df.dropna(subset=[label_col])
    if len(df) < n_before:
        st.warning(f"Dropped {n_before-len(df)} rows with missing labels after coercion.")
    df[label_col] = df[label_col].round().astype(int)

    st.write("Label distribution:")
    st.write(df[label_col].value_counts())

    # Vectorize
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["_clean_text_"])
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=random_state, stratify=y if len(np.unique(y))>1 else None)

    # store in session
    st.session_state["vectorizer"] = vectorizer
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
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
    if not all(k in st.session_state for k in ("vectorizer","X_train","X_test","y_train","y_test")):
        st.warning("Run 'Preprocess & Split' first.")
    else:
        X_train = st.session_state["X_train"]
        X_test = st.session_state["X_test"]
        y_train = st.session_state["y_train"]
        y_test = st.session_state["y_test"]

        with st.spinner("Training LogisticRegression..."):
            lr = LogisticRegression(max_iter=1000)
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

if st.button("Predict Sentences"):
    sentences = [s.strip() for s in input_text.split("\n") if s.strip()]
    if len(sentences) == 0:
        st.warning("Enter at least one non-empty sentence.")
    else:
        if "vectorizer" not in st.session_state:
            st.warning("You must run 'Preprocess & Split' (to fit vectorizer) and train models before prediction.")
        else:
            vectorizer = st.session_state["vectorizer"]
            X_new = vectorizer.transform([preprocess(s) for s in sentences])

            results = {"Sentence": sentences}
            # LogisticRegression
            if "lr_model" in st.session_state:
                lr = st.session_state["lr_model"]
                probs = lr.predict_proba(X_new)[:,1] if hasattr(lr, "predict_proba") else lr.decision_function(X_new)
                preds = lr.predict(X_new)
                results["LogReg_prob"] = [float(round(p,4)) for p in (probs.tolist() if hasattr(probs,'tolist') else probs)]
                results["LogReg_pred"] = ["Positive" if int(p)==1 else "Negative" for p in preds]
            else:
                results["LogReg_pred"] = ["(not trained)"]*len(sentences)

            # Naive Bayes
            if "nb_model" in st.session_state:
                nb = st.session_state["nb_model"]
                probs = nb.predict_proba(X_new)[:,1]
                preds = nb.predict(X_new)
                results["NB_prob"] = [float(round(p,4)) for p in probs.tolist()]
                results["NB_pred"] = ["Positive" if int(p)==1 else "Negative" for p in preds]
            else:
                results["NB_pred"] = ["(not trained)"]*len(sentences)

            # SVC
            if "svc_model" in st.session_state:
                svc = st.session_state["svc_model"]
                # SVC probability requires probability=True during training (we set it). If not available, fallback to decision_function
                if hasattr(svc, "predict_proba"):
                    probs = svc.predict_proba(X_new)[:,1]
                else:
                    probs = svc.decision_function(X_new)
                    # scale decision_function to 0..1 for display (sigmoid)
                    probs = 1/(1+np.exp(-probs))
                preds = svc.predict(X_new)
                results["SVC_prob"] = [float(round(p,4)) for p in probs.tolist()]
                results["SVC_pred"] = ["Positive" if int(p)==1 else "Negative" for p in preds]
            else:
                results["SVC_pred"] = ["(not trained)"]*len(sentences)

            res_df = pd.DataFrame(results)
            st.dataframe(res_df)

# ---- Save / Load model (optional) ----
st.subheader("Save / Load")
col1, col2 = st.columns(2)
with col1:
    if st.button("Save models & vectorizer to disk (pickle)"):
        import joblib, os
        if "vectorizer" not in st.session_state:
            st.warning("Nothing to save. Fit vectorizer and train models first.")
        else:
            out_dir = st.text_input("Output directory (relative)", value="model_artifacts")
            os.makedirs(out_dir, exist_ok=True)
            joblib.dump(st.session_state["vectorizer"], f"{out_dir}/vectorizer.joblib")
            for name in ("lr_model","nb_model","svc_model"):
                if name in st.session_state:
                    joblib.dump(st.session_state[name], f"{out_dir}/{name}.joblib")
            st.success(f"Saved artifacts to {out_dir}")

with col2:
    if st.button("Load models & vectorizer from disk (pickle)"):
        import joblib, os
        in_dir = st.text_input("Input directory (relative)", value="model_artifacts")
        try:
            st.session_state["vectorizer"] = joblib.load(f"{in_dir}/vectorizer.joblib")
            for name in ("lr_model","nb_model","svc_model"):
                path = f"{in_dir}/{name}.joblib"
                try:
                    st.session_state[name] = joblib.load(path)
                except Exception:
                    st.warning(f"Could not load {path}")
            st.success("Loaded artifacts (if present).")
        except FileNotFoundError:
            st.error("Directory or files not found.")

# ---- Notes ----
st.markdown("---")
st.markdown(
    """
- This app preprocesses text with lowercasing, punctuation & number removal, stopword filtering (keeps negations), and Porter stemming.
- Vectorization uses TF-IDF fitted during 'Preprocess & Split'.
- Train all three models using the 'Train' button. Predictions require the vectorizer + trained models in session.
- For large datasets, reduce TF-IDF dimension limits or train outside Streamlit.
"""
)


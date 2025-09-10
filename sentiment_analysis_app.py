# sentiment_analysis_app.py
import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import Callback
import tensorflow as tf

st.set_page_config(page_title="RNN Sentiment Classifier", layout="wide")

st.title("RNN — Text Sentiment Classifier (Streamlit)")

# ---------------------------
# Utility functions
# ---------------------------
def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_padded(texts, tokenizer, max_len):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post", value=0)

class StreamlitProgress(Callback):
    """Simple Keras callback to stream epochs progress to Streamlit"""
    def __init__(self, progress_bar, status_text):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.status_text.text(f"Epoch {epoch+1} — loss: {logs.get('loss', 0):.4f}  val_loss: {logs.get('val_loss', 0):.4f}  acc: {logs.get('accuracy', 0):.4f}")
        # update progress bar (assuming user sets epochs)
        self.progress_bar.progress(min((epoch+1)/self.params.get('epochs', 1), 1.0))

# ---------------------------
# Sidebar: data upload + params
# ---------------------------
st.sidebar.header("Data & Hyperparameters")

uploaded = st.sidebar.file_uploader("Upload dataset (CSV or TSV)", type=["csv","tsv","txt"])
sep_choice = st.sidebar.selectbox("Separator", options=[",", "\t"], index=0 if uploaded is None else (1 if uploaded.name.endswith(".tsv") else 0))
use_sample = st.sidebar.checkbox("Use bundled sample data instead", value=False)

if use_sample or uploaded is None:
    # sample small dataset
    sample = pd.DataFrame({
        "Review": [
            "I love this place",
            "Worst food ever",
            "Not bad could be better",
            "Absolutely fantastic experience",
            "Terrible service and boring food",
            "I really enjoyed it"
        ],
        "Liked": [1, 0, 1, 1, 0, 1]
    })
    df = sample
    st.sidebar.info("Using bundled sample data. Upload your file or uncheck this to use your file.")
else:
    try:
        df = pd.read_csv(uploaded, sep=sep_choice)
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded file: {e}")
        st.stop()

st.sidebar.markdown("### Column selection")
col_text = None
col_label = None
if not df.empty:
    cols = df.columns.tolist()
    col_text = st.sidebar.selectbox("Text column (review)", options=cols, index=0)
    col_label = st.sidebar.selectbox("Label column (binary 0/1)", options=cols, index=min(1, len(cols)-1))
else:
    st.sidebar.warning("No data loaded yet.")

st.sidebar.markdown("### Tokenizer / Model")
MAX_VOCAB = st.sidebar.number_input("Max vocabulary (num_words)", min_value=1000, max_value=50000, value=10000, step=500)
MAX_LEN = st.sidebar.number_input("Max sequence length (pad/truncate)", min_value=5, max_value=200, value=20, step=1)
EMBED_DIM = st.sidebar.number_input("Embedding dimension", min_value=8, max_value=256, value=32, step=8)
RNN_UNITS = st.sidebar.number_input("RNN units", min_value=8, max_value=256, value=32, step=8)
EPOCHS = st.sidebar.number_input("Epochs", min_value=1, max_value=50, value=8, step=1)
BATCH = st.sidebar.number_input("Batch size", min_value=8, max_value=512, value=32, step=8)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

# ---------------------------
# Show data preview and column mapping
# ---------------------------
st.subheader("Data preview")
st.write("Rows:", len(df))
st.dataframe(df.head(10))

if col_text is None or col_label is None:
    st.error("Please select both text and label columns in the sidebar.")
    st.stop()

# ---------------------------
# Preprocess & prepare
# ---------------------------
st.subheader("Prepare dataset")
with st.spinner("Cleaning and preparing data..."):
    # convert
    df[col_text] = df[col_text].astype(str).apply(clean_text)
    # try to coerce labels to int 0/1
    try:
        df[col_label] = pd.to_numeric(df[col_label], errors='coerce')
    except Exception:
        df[col_label] = df[col_label].astype(str).map(lambda x: 1 if str(x).lower() in ("1","yes","y","true","positive","pos") else 0)
    # drop rows with NaN in label
    n_before = len(df)
    df = df.dropna(subset=[col_label])
    n_after = len(df)
    if n_after < n_before:
        st.warning(f"Dropped {n_before-n_after} rows with missing labels after coercion.")

    # Ensure binary labels; if more than two unique values warn
    unique_labels = sorted(df[col_label].unique().tolist())
    if len(unique_labels) > 2:
        st.warning(f"Label column has more than 2 unique values: {unique_labels}. They will be coerced to 0/1 via rounding.")
        df[col_label] = df[col_label].round().astype(int)
    else:
        df[col_label] = df[col_label].astype(int)

st.write("Label distribution:")
st.write(df[col_label].value_counts())

# ---------------------------
# Tokenizer fit & dataset split
# ---------------------------
if st.button("Prepare tokenizer & split data"):
    X = df[col_text]
    y = df[col_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=int(random_state), stratify=y if len(y.unique())>1 else None)
    tok = Tokenizer(num_words=int(MAX_VOCAB), oov_token="<OOV>")
    tok.fit_on_texts(X_train)
    Xtr = to_padded(X_train, tok, int(MAX_LEN))
    Xte = to_padded(X_test, tok, int(MAX_LEN))
    ytr = y_train.values
    yte = y_test.values

    st.success("Tokenizer and split ready.")
    st.write("Training samples:", Xtr.shape[0], "Validation/Test samples:", Xte.shape[0])
    st.write("Vocabulary (fitted):", min(int(MAX_VOCAB), len(tok.word_index)+1))
    # store in session_state for later training/predict
    st.session_state['tok'] = tok
    st.session_state['Xtr'] = Xtr
    st.session_state['Xte'] = Xte
    st.session_state['ytr'] = ytr
    st.session_state['yte'] = yte

# ---------------------------
# Build & train model
# ---------------------------
st.subheader("Build & Train Model")
if 'tok' in st.session_state:
    if st.button("Build & Train model"):
        tok = st.session_state['tok']
        Xtr = st.session_state['Xtr']
        Xte = st.session_state['Xte']
        ytr = st.session_state['ytr']
        yte = st.session_state['yte']

        vocab_size = min(int(MAX_VOCAB), len(tok.word_index) + 1)
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=int(EMBED_DIM), input_length=int(MAX_LEN), mask_zero=True),
            SimpleRNN(int(RNN_UNITS)),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        st.write("Model summary:")
        model.summary(print_fn=lambda s: st.text(s))

        progress_bar = st.progress(0.0)
        status = st.empty()
        cb = StreamlitProgress(progress_bar, status)

        # Train
        history = model.fit(
            Xtr, ytr,
            epochs=int(EPOCHS),
            batch_size=int(BATCH),
            validation_split=0.2,
            callbacks=[cb],
            verbose=0
        )

        loss, acc = model.evaluate(Xte, yte, verbose=0)
        st.success(f"Test accuracy: {acc:.4f}   Test loss: {loss:.4f}")

        # save to session
        st.session_state['model'] = model
        st.session_state['history'] = history.history
else:
    st.info("Click 'Prepare tokenizer & split data' first.")

# ---------------------------
# Predictions / Demo
# ---------------------------
st.subheader("Predict — try examples or custom text")
if 'model' in st.session_state and 'tok' in st.session_state:
    model = st.session_state['model']
    tok = st.session_state['tok']

    # Multi-line input: one sentence per line
    st.markdown("**Custom Input (one sentence per line)**")
    user_input = st.text_area("Enter sentences (each on a new line)", 
                              value="The food was amazing\nThe service was bad")

    if st.button("Predict Sentences"):
        sentences = [line.strip() for line in user_input.split("\n") if line.strip()]
        if sentences:
            pads = to_padded([clean_text(s) for s in sentences], tok, int(MAX_LEN))
            probs = model.predict(pads, verbose=0).ravel()
            labels = ["positive" if p >= 0.5 else "negative" for p in probs]
            results = pd.DataFrame({
                "Sentence": sentences,
                "Probability": probs,
                "Prediction": labels
            })
            st.dataframe(results)
        else:
            st.warning("Please enter at least one sentence.")
else:
    st.info("Train a model first to enable prediction demo.")


# ---------------------------
# Optional: show training history
# ---------------------------
st.subheader("Training history (if available)")
if 'history' in st.session_state:
    hist = st.session_state['history']
    st.write(hist)
else:
    st.write("No history available.")

# ---------------------------
# Notes
# ---------------------------
st.markdown("---")
st.markdown("**Notes:**")
st.markdown("""
- Select the correct text and label columns from the sidebar; different datasets name columns differently.
- Labels are coerced to integers (0/1). Common true values `1, yes, y, true, positive` will be interpreted as 1.
- Training inside Streamlit runs in the foreground and may take time depending on data size and epochs. For large datasets, prefer training offline or reduce `MAX_LEN`, `EPOCHS`, and `BATCH`.
- This is a simple RNN demo for education and small datasets. For better performance use LSTM/GRU, dropout, regularization, or pretrained embeddings.
""")


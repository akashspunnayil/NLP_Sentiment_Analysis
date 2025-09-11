# Sentiment Analysis App (Streamlit + RNN + BiLSTM)

[View App](https://sentiment-nlp-app.streamlit.app)


A step-by-step guide for the **Sentiment Analysis (TF-IDF + ML + BiLSTM)** Streamlit app (`sentiment_analysis_app.py`).
This README explains the app purpose, how it works, how to run it locally and in the cloud, data expectations, UI workflow, and troubleshooting tips.

---

## Table of contents

1. Overview
2. Key features
3. Quick start (local)
4. Quick start (Streamlit Cloud)
5. Data format and examples
6. UI walk-through (step-by-step)
7. Preprocessing options and behavior
8. Models, training and evaluation
9. Prediction — line-by-line input
10. Save / Download / Load artifacts
11. NLTK data and bundling for deployment
12. Requirements and recommended Python version
13. Troubleshooting & common errors
14. Appendix: sample commands

---

## 1 — Overview

This app provides an interactive GUI for building simple NLP classifiers using TF-IDF features and classical ML models (Logistic Regression, Multinomial Naive Bayes, Linear SVC). It supports multi-class labels and gives per-class probabilities (when available). The UI covers dataset upload, column selection, optional preprocessing, train/test split, model training, evaluation, and per-sentence predictions.

---

## 2 — Key features

* Upload CSV / TSV or use the built-in sample dataset.
* Select the text column and label column from your file.
* Choose which label classes to include (multi-select).
* Option to enable/disable preprocessing (lowercase, punctuation removal, stopword filtering while keeping negations, Porter stemming).
* TF-IDF vectorization (fitted on the selected data).
* Train three models: LogisticRegression, MultinomialNB, SVC (linear).
* Multi-class support and per-class probability output.
* Download trained artifacts (vectorizer/models) to your local machine.
* Optional save/load to container filesystem (useful locally).

---

## 3 — Quick start (local)

Recommended Python: **3.11.x**

1. Create environment (conda example):

```bash
conda create -n nlp311 python=3.11 -y
conda activate nlp311
python -m pip install --upgrade pip
```

2. Install dependencies (example `requirements.txt`):

```
streamlit==1.38.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
joblib==1.5.2
nltk==3.9.0
```

Install:

```bash
pip install -r requirements.txt
```

3. (Optional) Download NLTK corpora locally if not bundling:

```bash
python -m nltk.downloader punkt stopwords wordnet omw-1.4
```

4. Run app:

```bash
streamlit run sentiment_analysis_app.py
```

Open the browser to the printed local URL.

---

## 4 — Quick start (Streamlit Cloud)

* If deploying to Streamlit Cloud, prefer **not** to rely on runtime `nltk.download()` (host may block downloads). Bundle minimal `nltk_data` (instructions below) or use the app’s built-in fallback (the app can be adapted to a local stopword list).
* Use the `requirements.txt` above (no TensorFlow unless you need it).
* Deploy from your GitHub repo. The app includes download buttons so you can fetch artifacts to your local machine.

---

## 5 — Data format and examples

### Minimal expectation

A tabular file with:

* one column containing the text (e.g., `Review`)
* one column containing labels (binary or multi-class; numeric or string)

### Example TSV (headers required)

```
Review    Liked
"I love this place"    1
"Worst food ever"      0
"Average experience"   2
```

### Notes

* The app discovers unique labels in the chosen label column and shows them in a sidebar multi-select; you can exclude classes before training.
* If your file uses tabs, select **Tab (\t)** as separator in the sidebar. The selectbox displays friendly labels (`Comma (,)`, `Tab (\t)`, `Semicolon (;)`).

---

## 6 — UI walk-through (step-by-step)

1. **Sidebar: Data & Settings**

   * Upload a CSV/TSV (or tick "Use built-in sample").
   * Choose the visible separator: Comma / Tab / Semicolon.
   * Pick Text column and Label column.
   * Pick which label classes to include (multi-select).
   * Toggle **Apply preprocessing to text** (default ON).
   * Tune test set fraction and random state.

2. **Data preview**

   * App shows first 10 rows and number of rows.

3. **Preprocess & Split**

   * Click **Preprocess & Split**.
   * The app filters rows by selected labels, optionally preprocesses text, fits TF-IDF on the chosen text, and performs train/test split.
   * Results: label distribution and confirmation that vectorizer and splits are stored in session.

4. **Train models**

   * Click **Train LogisticRegression, NaiveBayes, SVC**.
   * Each model is trained on the training split and evaluated on the test split. Accuracy and classification report are displayed. Models are saved to `st.session_state`.

5. **Predict line-by-line**

   * Enter one sentence per line in the main text area and click **Predict Sentences**.
   * App vectorizes input using the same TF-IDF and preprocessing flag used at fit time.
   * Output: a table with sentences, model predictions, and per-class probabilities (if available).

6. **Save / Download / Load**

   * Save writes `.joblib` artifacts into the container path (e.g., `model_artifacts/`) — useful for local development.
   * Download buttons let you download vectorizer and model files directly to your PC (works on Streamlit Cloud).
   * Load loads artifacts from the container filesystem back into session state (useful if artifacts are present locally or in the same running container).

---

## 7 — Preprocessing: what it does and how to control it

When **Apply preprocessing** is ON:

* Lowercases text
* Removes punctuation
* Removes numeric tokens
* Splits tokens on whitespace
* Removes English stopwords (except negations: `not`, `no`, `nor`, `n't`)
* Applies Porter stemming

If OFF:

* Raw string (string-cast) is used for TF-IDF. **Important**: vectorizer must be fitted in the same mode (preprocessed vs raw) that you later use for predictions. The app stores `apply_preprocess` in session and in saved metadata to enforce consistency.

---

## 8 — Models, training and evaluation

* **Vectorizer**: `TfidfVectorizer()` (fitted on dataset in the app; configurable extension possible: `max_features`, `ngram_range`, etc.)

* **Models**:

  * `LogisticRegression(max_iter=2000)` — supports multi-class by default
  * `MultinomialNB()` — common baseline for text
  * `SVC(kernel='linear', probability=True)` — provides `predict_proba` when `probability=True`
  * **BiLSTM (Bidirectional LSTM)** — deep learning sequence model trained on padded integer sequences  
    - **Tokenizer**: `Tokenizer(num_words=20000, oov_token="<OOV>")`  
    - **Architecture (binary)**:  
      `Embedding → Bidirectional(LSTM(128)) → Dropout → Dense(64, relu) → Dropout → Dense(1, sigmoid)`  
    - **Architecture (multi-class)**:  
      `Embedding → Bidirectional(LSTM(128)) → Dropout → Dense(64, relu) → Dropout → Dense(n_classes, softmax)`  
    - **Training**: `batch_size=64`, `epochs=6`, `EarlyStopping(patience=2)`  
    - **Evaluation**: reports precision, recall, f1-score, support (via `classification_report`) in addition to accuracy  
    - **Outputs**:  
      - `BiLSTM_pred` → predicted label (0/1 or class index)  
      - `BiLSTM_prob_*` → predicted probability per class (same style as scikit-learn models)


* **Evaluation**:

  * Accuracy and `classification_report` (precision/recall/f1 for each class)
  * For SVC, if `predict_proba` is not available, decision scores are converted into pseudo-probabilities (sigmoid for binary, softmax for multi-class).

---

## 9 — Prediction — details

* Input: one sentence per line.
* The app vectorizes each sentence using the TF-IDF fitted earlier. If preprocessing was enabled at fit time, predictions will preprocess input similarly.
* Output table includes:

  * model predictions (original label names),
  * per-class probability columns labeled like `LogReg_prob_{class}`.

---

## 10 — Save / Download / Load artifacts

* **Save to disk** (container): writes to `./model_artifacts/` (or folder set in sidebar). Useful locally; ephemeral on the cloud.
* **Download**: in-memory `joblib` dumps exposed via `st.download_button`. Use these to download artifacts to your laptop.
* **Load**: reads artifacts from container to `st.session_state`. Useful locally or when you commit artifacts to repository.

### Recommended workflow on Streamlit Cloud

1. Train models in the app.
2. Click each **Download** button to save `.joblib` files to your computer.
3. Optionally commit those files to your repo for later Load (only for small artifacts).

---

## 11 — NLTK data and bundling for deployment

**Problem**: `nltk.download()` can fail in restricted cloud environments. Solutions:

**A — Bundle `nltk_data` in your repo (recommended for cloud)**

1. Locally, run:

```bash
python - <<'PY'
import nltk, os
DATA_DIR = "nltk_data"
os.makedirs(DATA_DIR, exist_ok=True)
for r in ["punkt","stopwords","wordnet","omw-1.4"]:
    nltk.download(r, download_dir=DATA_DIR)
print("Done")
PY
```

2. Commit `nltk_data/` folder to repo.
3. At top of `sentiment_analysis_app.py` set:

```python
import os
HERE = os.path.dirname(__file__)
os.environ.setdefault("NLTK_DATA", os.path.join(HERE, "nltk_data"))
```

This ensures no runtime downloads are required.

**B — Use a small built-in stopword list instead of NLTK corpora**
If you only need stopwords and stemming, you can embed a small English stopword set in the code and only use `PorterStemmer()` from NLTK (Porter stemmer does not require corpora). This removes `nltk.download()` dependency.

---

## 12 — Requirements & recommended Python version

* Recommended Python: **3.11.x**
* Example `requirements.txt` for the TF-IDF + scikit-learn app:

```
streamlit==1.38.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
joblib==1.5.2
nltk==3.9.0
```

> If you include TensorFlow / deep learning parts, pick a TensorFlow + protobuf version compatible with Python 3.11 and pin versions accordingly.

---

## 13 — Troubleshooting & common errors

* **`LookupError` from NLTK**: bundle `nltk_data` or use builtin stopwords instead of runtime download. See section 11.
* **`DuplicateWidgetID`**: use explicit `key=` for widgets or avoid creating duplicate widgets with identical labels. The app uses explicit keys for Save/Load widgets.
* **Saved artifacts disappear after restart (cloud)**: container filesystem is ephemeral. Use the Download buttons (or external storage) to persist artifacts.
* **Incompatible TensorFlow/Python on deploy**: remove `tensorflow` from `requirements.txt` if not used. If TF is required, set app runtime to Python 3.11 and pin TF/protobuf versions.
* **`st.query_params` missing**: some streamlit versions don't expose `st.query_params`. Use `st.experimental_get_query_params()` or upgrade Streamlit.

---

## 14 — Appendix: sample commands

Create local environment and run:

```bash
conda create -n nlp311 python=3.11 -y
conda activate nlp311
pip install -r requirements.txt
streamlit run sentiment_analysis_app.py
```

Bundle NLTK data (local) and commit:

```bash
python - <<'PY'
import nltk, os
DATA_DIR='nltk_data'
os.makedirs(DATA_DIR, exist_ok=True)
for r in ['punkt','stopwords','wordnet','omw-1.4']:
    nltk.download(r, download_dir=DATA_DIR)
PY
git add nltk_data
git commit -m "Add bundled NLTK data"
git push
```

Download trained artifacts from the app (click buttons in UI) and load locally:

```bash
# After downloading artifact files locally to ./model_artifacts/
streamlit run sentiment_analysis_app.py
# in the app click "Load models & vectorizer from disk"
```

---

## Closing notes

* The app is intentionally lightweight and designed for quick iteration and demonstration of classic NLP pipelines.
* If you need reproducible production inference, export trained artifacts using the download buttons and host inference in a separate service (Flask/FastAPI) with pinned environments and persistent storage (S3/GCS).

If you’d like, I can:

* Generate a `README.md` file content ready to paste into your repo.
* Produce a `requirements.txt` and `runtime.txt` (Python 3.11) for deployment.
* Produce the `nltk_data` creation script and a small `.gitignore` suggestion for large files.


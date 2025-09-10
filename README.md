# Sentiment Analysis App (Streamlit + RNN)

[View App](https://sentiment-nlp-app.streamlit.app)

This Streamlit app uses a SimpleRNN (Recurrent Neural Network) to perform sentiment classification on text reviews. It provides a user-friendly interface for training, evaluating, and testing sentiment predictions on custom datasets.

## Objective

To classify text reviews (e.g., restaurant reviews) as **positive** or **negative** using a deep learning RNN model with an interactive Streamlit interface.

## Workflow

- Upload a dataset (CSV/TSV) with **text** and **label** columns  
- Select the appropriate columns from the sidebar (e.g., `Review`, `Liked`)  
- Clean and preprocess the text data  
- Tokenize and pad sequences  
- Train a SimpleRNN model with embedding layer  
- Evaluate model accuracy  
- Test predictions on sample sentences or user input (line-by-line)

## Features

- Upload your own dataset (CSV/TSV)  
- Flexible column selection for text and label  
- Text cleaning and preprocessing  
- Tokenizer and sequence padding with adjustable parameters  
- Interactive RNN model training and evaluation  
- Prediction on multiple sentences entered line by line  
- Built-in example test sentences for quick demo  

## Dependencies

- `streamlit`  
- `pandas`  
- `numpy`  
- `scikit-learn`  
- `tensorflow`  

## Output

- Model training summary and accuracy metrics  
- Label distribution overview  
- Interactive prediction results for input sentences  
- Table output with predicted probabilities and sentiment labels  



import streamlit as st
import os
import joblib
import pickle
import numpy as np
import pandas as pd
import logging
import warnings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from streamlit_option_menu import option_menu
import plotly.graph_objects as go

# -----------------------------
# CONFIGURATION & LOGGING
# -----------------------------
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# -----------------------------
# STYLES & NAVIGATION
# -----------------------------
def init_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
        .main-header { font-size: 3rem; text-align: center; margin-bottom: 2rem; color: #DA70D6; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
        .ai-pred { background-color: #FECACA; color: #C0392B; padding: 1.5rem; border-radius: 1.5rem; margin-bottom: 1rem; text-align: center; font-size: 1.5rem; font-weight: 600; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .human-pred { background-color: #A7F3D0; color: #27AE60; padding: 1.5rem; border-radius: 1.5rem; margin-bottom: 1rem; text-align: center; font-size: 1.5rem; font-weight: 600; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)


def menu():
    return option_menu(
        menu_title="🌸 AI vs Human Detector ✨",
        options=["Home","Single Detection","Batch Processing","Model Comparison","Help"],
        icons=["house-fill","magic","files","bar-chart-fill","question-circle-fill"],
        menu_icon="robot",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#FFFFFF"},
            "icon": {"color": "#DA70D6", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "--hover-color": "#FFF0F5"},
            "nav-link-selected": {"background-color": "#FFB6C1", "color": "#4B0082"}
        }
    )

# -----------------------------
# MODEL LOADING & PREDICTION
# -----------------------------
def load_all_models_debug():
    models = {}
    MODEL_DIR = 'models'
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        logging.warning(f"Created '{MODEL_DIR}' directory")
    # TF-IDF Vectorizer
    tfidf_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    models['tfidf_available'] = os.path.exists(tfidf_path)
    if models['tfidf_available']:
        models['tfidf_vectorizer'] = joblib.load(tfidf_path)
    # Tokenizer
    tok_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')
    models['tokenizer_available'] = os.path.exists(tok_path)
    if models['tokenizer_available']:
        models['tokenizer'] = pickle.load(open(tok_path,'rb'))
    # Model configurations with correct paths
    configs = {
        'svm': { 'type':'sklearn', 'path': os.path.join(MODEL_DIR,'svm_model.pkl'), 'name':'Support Vector Machine'},
        'decision_tree': { 'type':'sklearn','path': os.path.join(MODEL_DIR,'decision_tree_model.pkl'), 'name':'Decision Tree'},
        'adaboost': { 'type':'sklearn','path': os.path.join(MODEL_DIR,'adaboost_model.pkl'), 'name':'AdaBoost'},
        'cnn': { 'type':'keras', 'meta': os.path.join(MODEL_DIR,'CNN.pkl'), 'name':'Convolutional Neural Network'},
        'lstm': { 'type':'keras', 'meta': os.path.join(MODEL_DIR,'LSTM.pkl'), 'name':'Long Short-Term Memory'},
        'rnn': { 'type':'keras', 'meta': os.path.join(MODEL_DIR,'RNN.pkl'), 'name':'Recurrent Neural Network'}
    }
    # Load each model
    for key, cfg in configs.items():
        models[f'{key}_config'] = cfg
        models[f'{key}_available'] = False
        if cfg['type'] == 'sklearn' and os.path.exists(cfg['path']):
            try:
                models[key] = joblib.load(cfg['path'])
                models[f'{key}_available'] = True
            except Exception as e:
                logging.error(f"Failed to load {key}: {e}")
        elif cfg['type'] == 'keras' and os.path.exists(cfg['meta']):
            try:
                meta = joblib.load(cfg['meta'])
                km = load_model(meta['model_path'])
                models[key] = km
                models[f'{key}_max_length'] = meta.get('max_length',100)
                models[f'{key}_available'] = True
            except Exception as e:
                logging.error(f"Failed to load {key}: {e}")
    return models


def get_prediction_explanation(text, key, prob, cfg):
    words = text.split()
    feats = {'word_count': len(words), 'char_count': len(text)}
    base = f"Confidence: {prob:.1%} | Words: {feats['word_count']} | Chars: {feats['char_count']}"
    expls = {
        'svm': 'SVM on TF-IDF. ',
        'decision_tree': 'Decision Tree splits. ',
        'adaboost': 'AdaBoost ensemble. ',
        'cnn': 'CNN convolution patterns. ',
        'lstm': 'LSTM sequential memory. ',
        'rnn': 'RNN sequence processing. '
    }
    return expls.get(key, 'Model analysis. ') + base


def make_prediction_debug(text, key, models):
    try:
        if not text.strip(): return None, None, 'Empty input'
        cfg = models.get(f'{key}_config', {})
        if not models.get(f'{key}_available', False): return None, None, 'Model not available'
        probs = None
        if cfg['type'] == 'sklearn':
            if not models['tfidf_available']: return None, None, 'No TF-IDF'
            X = models['tfidf_vectorizer'].transform([text]).toarray()
            exp_feat = getattr(models[key], 'n_features_in_', None)
            if exp_feat and X.shape[1] != exp_feat:
                diff = exp_feat - X.shape[1]
                X = np.hstack([X, np.zeros((1, max(diff, 0)))])[:,:exp_feat]
            if hasattr(models[key], 'predict_proba'):
                probs = models[key].predict_proba(X)[0]
            else:
                dec = models[key].decision_function(X)[0]
                p = 1 / (1 + np.exp(-dec))
                probs = np.array([1 - p, p])
        else:
            tok = models['tokenizer'] if models['tokenizer_available'] else Tokenizer(num_words=10000, oov_token='<OOV>')
            if not models['tokenizer_available']: tok.fit_on_texts([text])
            seq = tok.texts_to_sequences([text])
            pad = pad_sequences(seq, maxlen=models.get(f'{key}_max_length', 100), padding='post')
            pred = models[key].predict(pad, verbose=0)
            if pred.shape[-1] > 1:
                probs = pred[0]
            else:
                ai = float(pred[0][0])
                probs = np.array([1 - ai, ai])
        if probs is None: return None, None, 'Prediction failed'
        probs = probs / np.sum(probs)
        idx = int(np.argmax(probs))
        label = ['Human', 'AI'][idx]
        return label, probs, get_prediction_explanation(text, key, probs[idx], cfg)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None, None, f"Error: {e}"

# -----------------------------
# MAIN APPLICATION
# -----------------------------
def main():
    st.set_page_config(page_title="🌸 AI vs Human Detector ✨", page_icon="💖", layout="wide")
    init_css()
    sel = menu()
    with st.spinner('Loading models...'):
        models = load_all_models_debug()
    available = [(k, models[f'{k}_config']['name']) for k in ['svm','decision_tree','adaboost','cnn','lstm','rnn'] if models.get(f'{k}_available', False)]
    if not available:
        st.error('❌ No models available. Ensure model files are in `models/`.')
        return

    if sel == 'Home':
        st.markdown('<h1 class="main-header">🌸 AI vs Human Detector ✨</h1>', unsafe_allow_html=True)
        names = [name for _, name in available]
        st.info(f"💖 **Loaded Models:** {', '.join(names)}")
        st.markdown('Use the sidebar to select: Single Detection, Batch Processing, Model Comparison, or Help.')

    elif sel == 'Single Detection':
        st.header('✨ Single Text Analysis ✨')
        key = st.selectbox('Choose Model', available, format_func=lambda x: x[1])[0]
        text = st.text_area('Enter text to analyze', height=200)
        if st.button('Analyze'):
            label, probs, expl = make_prediction_debug(text, key, models)
            if label:
                cls = 'ai-pred' if label == 'AI' else 'human-pred'
                st.markdown(f"<div class='{cls}'><strong>{label}</strong> ({max(probs):.1%})</div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.metric('Human', f"{probs[0]:.1%}")
                c2.metric('AI', f"{probs[1]:.1%}")
                st.info(expl)
            else:
                st.error(expl)

    elif sel == 'Batch Processing':
        st.header('✨ Batch Text Analysis ✨')
        uploaded = st.file_uploader('Upload .txt or .csv', type=['txt','csv'])
        key = st.selectbox('Choose Model', available, format_func=lambda x: x[1])[0]
        if st.button('Process') and uploaded:
            if uploaded.type == 'text/plain':
                texts = uploaded.getvalue().decode('utf-8').splitlines()
            else:
                df = pd.read_csv(uploaded)
                texts = df.iloc[:, 0].astype(str).tolist()
            results, prog = [], st.progress(0)
            for i, t in enumerate(texts):
                lbl, probs, _ = make_prediction_debug(t, key, models)
                results.append({'Text': t, 'Prediction': lbl, 'Human': f"{probs[0]:.1%}", 'AI': f"{probs[1]:.1%}"})
                prog.progress((i+1)/len(texts))
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            st.download_button('Download CSV', pd.DataFrame(results).to_csv(index=False), mime='text/csv')

    elif sel == 'Model Comparison':
        st.header('✨ Compare Models ✨')
        text = st.text_area('Enter text for comparison', height=150)
        if st.button('Compare') and text:
            comp = []
            for key, name in available:
                lbl, probs, _ = make_prediction_debug(text, key, models)
                comp.append({'Model': name, 'Prediction': lbl, 'Human': probs[0], 'AI': probs[1]})
            df = pd.DataFrame(comp)
            st.table(df[['Model', 'Prediction']])
            fig = go.Figure(data=[
                go.Bar(name='Human', x=df['Model'], y=df['Human']),
                go.Bar(name='AI', x=df['Model'], y=df['AI'])
            ])
            fig.update_layout(barmode='group', title='Probability by Model')
            st.plotly_chart(fig, use_container_width=True)

    elif sel == 'Help':
        st.header('💖 Help & Instructions 💖')
        st.markdown("""
        * **Home**: View which models are loaded and ready (SVM, Decision Tree, AdaBoost, CNN, LSTM, RNN).
        * **Single Detection**: Analyze one piece of text at a time with any model.
        * **Batch Processing**: Upload a file (.txt or .csv) to run bulk predictions.
        * **Model Comparison**: See side-by-side predictions and probabilities across all models.
        * **Models Supported**:
          - **SVM**: Support Vector Machine on TF-IDF features
          - **Decision Tree**: Rule-based splits on TF-IDF
          - **AdaBoost**: Ensemble of weak learners on TF-IDF
          - **CNN/LSTM/RNN**: Deep learning models with learned embeddings
        * Ensure your model files (`.pkl`, `.h5`, etc.) and tokenizers are placed in the `models/` directory.
        """)

if __name__ == '__main__':
    main()

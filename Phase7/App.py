import streamlit as st
import numpy as np
import pickle
import os

# ---------------------------------------------------------
# CUSTOM MODEL CLASS (Required for unpickling)
# ---------------------------------------------------------
class MyMultinomialNB:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self._priors = np.zeros(len(self.classes))
        self._likelihoods = np.zeros((len(self.classes), n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self._priors[idx] = X_c.shape[0] / n_samples
            self._likelihoods[idx, :] = (X_c.sum(axis=0) + 1) / (X_c.sum() + n_features)

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self._priors[idx])
            likelihood = np.sum(np.log(self._likelihoods[idx, :]) * x)
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]

# ---------------------------------------------------------
# UI CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Spam Detector", page_icon="✉️")

st.markdown("""
<style>
    .stButton>button { width: 100%; height: 3em; background-color: #4CAF50; color: white; border-radius: 8px; }
    .stTextArea>div>div>textarea { background-color: #ffffff; color: #000000; border: 1px solid #ccc; }
</style>
""", unsafe_allow_html=True)

st.title("✉️ Simple Spam Detector")
st.caption("Powered by: Naïve Bayes (From Scratch)")

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(MODEL_DIR, "nb_scratch_model.pkl")
        vec_path = os.path.join(MODEL_DIR, "phase2_vectorizer.pkl")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vec_path, 'rb') as f:
            vec = pickle.load(f)
        return model, vec
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, vectorizer = load_model()

# ---------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------
msg = st.text_area("Message to Check:", height=150, placeholder="Paste your email or SMS here...")

if st.button("Check for Spam"):
    if not msg.strip():
        st.warning("Please enter some text.")
    elif model and vectorizer:
        # Preprocess using loaded vectorizer
        vec_text = vectorizer.transform([msg]).toarray()
        
        # Predict
        prediction = model.predict(vec_text)[0]
        
        st.write("---")
        if prediction == 1:
            st.error("🚨 RESULT: THIS IS SPAM!")
            st.write("**Be careful!** Do not click links or reply.")
        else:
            st.success("✅ RESULT: THIS IS HAM (SAFE)")
            
    else:
        st.error("Model could not be loaded.")
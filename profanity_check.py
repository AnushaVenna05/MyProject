import importlib.resources
import numpy as np
import joblib

# Assuming your package structure is like this:
# profanity_check/
#     data/
#         vectorizer.joblib
#         model.joblib

with importlib.resources.path('profanity_check.data', 'vectorizer.joblib') as vectorizer_path:
    vectorizer = joblib.load(vectorizer_path)

with importlib.resources.path('profanity_check.data', 'model.joblib') as model_path:
    model = joblib.load(model_path)

def _get_profane_prob(prob):
    return prob[1]

def predict(texts):
    return model.predict(vectorizer.transform(texts))

def predict_prob(texts):
    return np.apply_along_axis(_get_profane_prob, 1, model.predict_proba(vectorizer.transform(texts)))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Read in data
data = pd.read_csv('clean_data.csv')
texts = data['text'].astype(str)
y = data['is_offensive']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', min_df=0.0001)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_vectorized, y_train)

# Evaluate model
train_accuracy = model.score(X_train_vectorized, y_train)
test_accuracy = model.score(X_test_vectorized, y_test)
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Save the model and vectorizer
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(model, 'model.joblib')

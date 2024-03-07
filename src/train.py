import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib

file_path = '../datasets/train.csv'
df = pd.read_csv(file_path)
df.head()

df = pd.read_csv(file_path)

# Preprocess the data
X = df['url']
y = df['label']

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Define the sub models
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
knn_clf = KNeighborsClassifier(n_neighbors=5)
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
lr_clf = LogisticRegression(random_state=42)
#svm_clf = SVC(probability=True, random_state=42)
nb_clf = MultinomialNB()
dt_clf = DecisionTreeClassifier(random_state=42)
ada_clf = AdaBoostClassifier(n_estimators=50, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Use TfidfVectorizer for text vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)


# Create pipelines for each algorithm
rf_pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('classifier', rf_clf),
])

knn_pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('classifier', knn_clf),
])

lr_pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('classifier', lr_clf),
])
xgb_pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('classifier', xgb_clf),
])
'''
svm_pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('classifier', svm_clf),
])
'''
nb_pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('classifier', nb_clf),
])

dt_pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('classifier', dt_clf),
])

ada_pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('classifier', ada_clf),
])

gb_pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('classifier', gb_clf),
])

# Define the VotingClassifier with soft voting
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_pipeline),
        ('knn', knn_pipeline),
        ('xgb', xgb_pipeline),
        ('lr', lr_pipeline),
        ('nb', nb_pipeline),
        ('dt', dt_pipeline),
        ('ada', ada_pipeline),
        ('gb', gb_pipeline),
    ],
    voting='soft',
    verbose=True
)

# Train the model on the full dataset, assuming we're focusing on enhancing accuracy without a train-test split
voting_clf.fit(X, y_encoded)

# Save the model and the label encoder for future use
joblib.dump(voting_clf, 'model.joblib')
joblib.dump(le, 'label_encoder.joblib')


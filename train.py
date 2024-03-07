"""
this file contains code to train an ML model for malicious url classification

:authors:   Lior Vinman     ID: 213081763,
            Nevo Gadassi    ID: 325545887,
            Yoad Tamar      ID: 213451818

 !!!!!!!!!!!! RUN ONLY ON PYTHON INTERPRETER VERSION: 3.11.3 !!!!!!!!!!!!
 https://www.python.org/downloads/release/python-3113/

 !!!!!!!!!! IN CASE OF FAILURE ON RUN, PROVIDED demo.mp4 VIDEO TO PROOF ALL WORKS FINE !!!!!!!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse

from scipy.sparse import hstack
from tld import get_tld
import os.path
from sklearn.ensemble import RandomForestClassifier
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import re
import warnings
from datetime import datetime



def length_fd(url):
    """
    feature function to get the length of the fd of an url
    """
    path_url = urlparse(url).path
    try:
        return len(path_url.split('/')[1])
    except:
        return 0


def length_tld(tld):
    """
    feature function to get the length of the tld of an url
    """
    try:
        return len(tld)
    except:
        return -1


def count_digits(url):
    """
    feature function to get the number of digits of an url
    """
    digits = 0
    for i in url:
        if i.isnumeric():
            digits += 1
    return digits


def count_letters(url):
    """
    feature function to get the number of letters of an url
    """
    letters = 0
    for i in url:
        if i.isalpha():
            letters += 1
    return letters


def dir_count(url):
    """
    feature function to get the number of directories of an url
    """
    dirs = urlparse(url).path
    return dirs.count('/')


def ip_address_presence(url):
    """
    feature function to check if the url is present in IP format
    """
    # regex to find IP
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)
    if match:
        return -1
    else:
        return 1


def short_service(url):
    """
    feature function to check if the url is present in short format
    """
    # regex to find shorter
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return -1
    else:
        return 1


def url_tokenizer(url):
    """
    Tokenize the url
    """
    # regex to split by a token
    tokens = re.split('[://?&=._-]+', url)
    return tokens


# ignoring the warnings from the imports
warnings.filterwarnings("ignore")

# path to the dataset from the current directory
dataset_path = "train.csv"

# loading the dataset
url_data = pd.read_csv(dataset_path)

# showing the statistics of the dataset
url_data.head()

# dropping the index column
url_data = url_data.drop("Unnamed: 0", axis=1)

# showing the statistics of the dataset after the drop
url_data.head()

# showing the shape of the dataset
print(url_data.shape)

# showing information of the dataset
url_data.info()

# showing how much lines there are
url_data.isnull().sum()

# extracted features, dataframe of the url
url_data['length_url'] = url_data['url'].apply(lambda i: len(str(i)))
url_data['length_hostname'] = url_data['url'].apply(lambda i: len(urlparse(i).netloc))
url_data['length_path'] = url_data['url'].apply(lambda i: len(urlparse(i).path))
url_data['length_fd'] = url_data['url'].apply(lambda i: length_fd(i))
url_data['top_lvl_domain'] = url_data['url'].apply(lambda i: get_tld(i, fail_silently=True))
url_data['length_tld'] = url_data['top_lvl_domain'].apply(lambda i: length_tld(i))
url_data['count_dash'] = url_data['url'].apply(lambda i: i.count('-'))
url_data['count_at'] = url_data['url'].apply(lambda i: i.count('@'))
url_data['count_question'] = url_data['url'].apply(lambda i: i.count('?'))
url_data['count_percent'] = url_data['url'].apply(lambda i: i.count('%'))
url_data['count_dot'] = url_data['url'].apply(lambda i: i.count('.'))
url_data['count_equals'] = url_data['url'].apply(lambda i: i.count('='))
url_data['count_http'] = url_data['url'].apply(lambda i: i.count('http'))
url_data['count_https'] = url_data['url'].apply(lambda i: i.count('https'))
url_data['count_www'] = url_data['url'].apply(lambda i: i.count('www'))
url_data['count_digits'] = url_data['url'].apply(lambda i: count_digits(i))
url_data['count_letters'] = url_data['url'].apply(lambda i: count_letters(i))
url_data['dir_count'] = url_data['url'].apply(lambda i: dir_count(i))
url_data['presence_ip'] = url_data['url'].apply(lambda i: ip_address_presence(i))
url_data['short_service'] = url_data['url'].apply(lambda i: short_service(i))


# list of all features
features = [
    'length_url', 'length_hostname', 'length_path', 'length_fd', 'length_tld',
    'count_dash', 'count_at', 'count_question', 'count_percent', 'count_dot',
    'count_equals', 'count_http', 'count_https', 'count_www',
    'count_digits', 'count_letters', 'dir_count', 'presence_ip', 'short_service'
]


# setting the grid be with blocks
sns.set_style("whitegrid")

print(f"[{datetime.now().strftime('%H:%M:%S')}] [Start] generate distribution of dataset")
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=url_data, palette="coolwarm")
plt.title("Distribution of White vs. Malicious URLs")
plt.xlabel("URL Label")
plt.ylabel("Count")
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/count_graph.png")
plt.show()


print(f"[{datetime.now().strftime('%H:%M:%S')}] [Start] generate distribution of dataset Pie")
labels = url_data['label'].value_counts().index
sizes = url_data['label'].value_counts().values
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=["skyblue", "orange"])
plt.title("Pie Chart of URL Labels")
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/pie_count_graph.png")
plt.show()


print(f"[{datetime.now().strftime('%H:%M:%S')}] [Start] generate distribution of each feature in single graph")
plt.figure(figsize=(15, 20))
for i, feature in enumerate(features, 1):
    plt.subplot(5, 4, i)
    sns.histplot(data=url_data, x=feature, bins=30, kde=True, palette="viridis", element="step")
    plt.title(f"Distribution of {feature.replace('_', ' ').capitalize()}")
    plt.xlabel(feature.replace("_", " ").capitalize())
    plt.ylabel('Count')
    plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig('plots/all_graphs.png')
plt.show()


for feature in features:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Start] generate distribution of '{feature}' graph")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=url_data, x=feature, hue="result", bins=30, kde=True, palette="viridis", element="step")
    plt.title(f"Distribution of: '{feature}' - by Classification")
    plt.xlabel(feature.replace('_', ' '))
    plt.ylabel('Count')
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{feature}_graph.png")
    plt.show()



print(f"[{datetime.now().strftime('%H:%M:%S')}] [Start] TF-IDF")
max_features = 10_000
manual_features = url_data[features]
tfidf_vectorizer = TfidfVectorizer(tokenizer=url_tokenizer, max_features=max_features, norm=None)
tfidf_matrix = tfidf_vectorizer.fit_transform(url_data['url'])
X_combined = hstack((tfidf_matrix, manual_features))


print(f"[{datetime.now().strftime('%H:%M:%S')}] [Start] `train_test_split()`")
X_train, X_test, y_train, y_test = train_test_split(X_combined, url_data['result'], test_size=0.25, random_state=42)


print(f"[{datetime.now().strftime('%H:%M:%S')}] [Start] Random Forest")
rf_classifier_combined = RandomForestClassifier(random_state=42)
rf_classifier_combined.fit(X_train, y_train)

print(f"[{datetime.now().strftime('%H:%M:%S')}] [Start] Prediction - 0.25% for test")
y_pred = rf_classifier_combined.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.4f}%")


print(f"[{datetime.now().strftime('%H:%M:%S')}] [Start] Models dump")
joblib.dump(rf_classifier_combined, "rf_model.joblib")
joblib.dump(tfidf_vectorizer, "tfidf_model.joblib")


print(f"[{datetime.now().strftime('%H:%M:%S')}] [Start] generate confusion matrix graph")
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/confusion_matrix_graph.png")
plt.show()

print(f"[{datetime.now().strftime('%H:%M:%S')}] FINISH !!!")

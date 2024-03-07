import os
import re
import joblib
from urllib.parse import urlparse
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from tld import get_tld
import pandas as pd
import warnings

# ignoring the warnings from the imports
warnings.filterwarnings("ignore")


def length_fd(url):
    try:
        path_url = urlparse(url).path
        return len(path_url.split('/')[1])
    except:
        return 0


def length_tld(tld):
    try:
        return len(tld)
    except:
        return -1


def count_digits(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits += 1
    return digits


def count_letters(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters += 1
    return letters


def dir_count(url):
    dirs = urlparse(url).path
    return dirs.count('/')


def ip_address_presence(url):
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
    tokens = re.split('[://?&=._-]+', url)
    return tokens


def extract_features(url):
    """
    this function extracts our features from the given url
    """
    parsed_url = urlparse(url)
    features = {
        'length_url': len(url),
        'length_hostname': len(parsed_url.netloc),
        'length_path': len(parsed_url.path),
        'length_fd': length_fd(url),
        'length_tld': length_tld(get_tld(url, fail_silently=True)),
        'top_lvl_domain': get_tld(url, fail_silently=True),
        'count_dash': url.count('-'),
        'count_at': url.count('@'),
        'count_question': url.count('?'),
        'count_percent': url.count('%'),
        'count_dot': url.count('.'),
        'count_equals': url.count('='),
        'count_http': url.count('http'),
        'count_https': url.count('https'),
        'count_www': url.count('www'),
        'count_digits': count_digits(url),
        'count_letters': count_letters(url),
        'dir_count': dir_count(url),
        'presence_ip': ip_address_presence(url),
        'short_service': short_service(url)
    }
    # returning dataframe with the features
    return pd.DataFrame([features])


def check_url(url):
    """
    this function checks if the url is malicious using our trained ML model
    """

    # Load the trained RandomForestClassifier model
    rf_model = joblib.load("rf_model.joblib")
    tfidf_model = joblib.load("tfidf_model.joblib")

    # Extract manual features
    length_url = len(url)
    length_hostname = len(urlparse(url).netloc)
    length_path = len(urlparse(url).path)
    length_fd_val = length_fd(url)
    top_lvl_domain = get_tld(url, fail_silently=True)
    length_tld_val = length_tld(top_lvl_domain)
    count_dash = url.count('-')
    count_at = url.count('@')
    count_question = url.count('?')
    count_percent = url.count('%')
    count_dot = url.count('.')
    count_equals = url.count('=')
    count_http = url.count('http')
    count_https = url.count('https')
    count_www = url.count('www')
    count_digits_val = count_digits(url)
    count_letters_val = count_letters(url)
    dir_count_val = dir_count(url)
    presence_ip_val = ip_address_presence(url)
    short_service_val = short_service(url)

    # Combine manual features
    manual_features = [length_url, length_hostname, length_path, length_fd_val, length_tld_val,
                       count_dash, count_at, count_question, count_percent, count_dot,
                       count_equals, count_http, count_https, count_www,
                       count_digits_val, count_letters_val, dir_count_val, presence_ip_val, short_service_val]

    # Combine manual features into a single NumPy array
    new_url_manual_features = np.array(manual_features).reshape(1, -1)

    # Use the TF-IDF vectorizer on the new URL
    new_url_tfidf_matrix = tfidf_model.transform([url])

    # Combine TF-IDF and manual features
    new_url_combined_features = hstack((new_url_tfidf_matrix, new_url_manual_features))

    # Make predictions using the combined features
    prediction = rf_model.predict(new_url_combined_features)

    # return the prediction, 0 = white, 1 = malicious
    return prediction[0]

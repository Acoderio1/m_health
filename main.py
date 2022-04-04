import requests
import pickle
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from math import log, sqrt
import http.client
import json
import csv
import datetime

from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/')
def hello_world():
    json_file = {}
    json_file['query'] = 'hello_world'
    return jsonify(json_file)


@app.route('/instadepml/<id>')
def get_world(id):
    conn = http.client.HTTPSConnection("simple-instagram-scraper1.p.rapidapi.com")

    headers = {
        'x-rapidapi-host': "simple-instagram-scraper1.p.rapidapi.com",
        'x-rapidapi-key': "e056db6af5msh5b87eb4727d8928p18cdddjsn6638d71b971c"
    }

    conn.request("GET", "/api/profile/"+id+"/posts", headers=headers)

    res = conn.getresponse()
    data = res.read()

    result = data.decode("utf-8")
    json_object = json.loads(result)
    json_formatted_str = json.dumps(json_object, indent=2)

    # print(json_formatted_str)
    json_object = json.dumps(json_formatted_str, indent=4)
    with open('readmedf.json', 'w') as f:
        f.write(json_formatted_str)

    f = open('readmedf.json')
    x = json.load(f)
    c = 0
    a = []
    for i in x:
        c = c + 1
        print(i['title'])
        a.append(i['title'])
    # print(c)

    tweetsx = a
    f.close()

    tweets = pd.read_csv('sentiment_tweets3.csv')
    tweets.head(20)

    tweets.drop(['Unnamed: 0'], axis=1, inplace=True)

    tweets['label'].value_counts()

    tweets.info()

    """# Splitting the Data in Training and Testing Sets

    As you can see, I used almost all the data for training: 98% and the rest for testing.
    """

    totalTweets = 8000 + 2314
    trainIndex, testIndex = list(), list()
    for i in range(tweets.shape[0]):
        if np.random.uniform(0, 1) < 0.98:
            trainIndex += [i]
        else:
            testIndex += [i]
    trainData = tweets.iloc[trainIndex]
    testData = tweets.iloc[testIndex]

    tweets.info()

    trainData['label'].value_counts()

    trainData.head()

    testData['label'].value_counts()

    testData.head()

    """# Wordcloud Analysis"""

    depressive_words = ' '.join(list(tweets[tweets['label'] == 1]['message']))

    positive_words = ' '.join(list(tweets[tweets['label'] == 0]['message']))

    """#Pre-processing the data for the training: Tokenization, stemming, and removal of stop words"""

    def process_message(message, lower_case=True, stem=True, stop_words=True, gram=2):
        if lower_case:
            message = message.lower()
        words = word_tokenize(message)
        words = [w for w in words if len(w) > 2]
        if gram > 1:
            w = []
            for i in range(len(words) - gram + 1):
                w += [' '.join(words[i:i + gram])]
            return w
        if stop_words:
            sw = stopwords.words('english')
            words = [word for word in words if word not in sw]
        if stem:
            stemmer = PorterStemmer()
            words = [stemmer.stem(word) for word in words]
        return words

    class TweetClassifier(object):
        def __init__(self, trainData, method='tf-idf'):
            self.tweets, self.labels = trainData['message'], trainData['label']
            self.method = method

        def train(self):
            self.calc_TF_and_IDF()
            if self.method == 'tf-idf':
                self.calc_TF_IDF()
            else:
                self.calc_prob()

        def calc_prob(self):
            self.prob_depressive = dict()
            self.prob_positive = dict()
            for word in self.tf_depressive:
                self.prob_depressive[word] = (self.tf_depressive[word] + 1) / (self.depressive_words + \
                                                                               len(list(self.tf_depressive.keys())))
            for word in self.tf_positive:
                self.prob_positive[word] = (self.tf_positive[word] + 1) / (self.positive_words + \
                                                                           len(list(self.tf_positive.keys())))
            self.prob_depressive_tweet, self.prob_positive_tweet = self.depressive_tweets / self.total_tweets, self.positive_tweets / self.total_tweets

        def calc_TF_and_IDF(self):
            noOfMessages = self.tweets.shape[0]
            self.depressive_tweets, self.positive_tweets = self.labels.value_counts()[1], self.labels.value_counts()[0]
            self.total_tweets = self.depressive_tweets + self.positive_tweets
            self.depressive_words = 0
            self.positive_words = 0
            self.tf_depressive = dict()
            self.tf_positive = dict()
            self.idf_depressive = dict()
            self.idf_positive = dict()
            for i in range(noOfMessages):
                message_processed = process_message(self.tweets.iloc[i])
                count = list()  # To keep track of whether the word has ocured in the message or not.
                # For IDF
                for word in message_processed:
                    if self.labels.iloc[i]:
                        self.tf_depressive[word] = self.tf_depressive.get(word, 0) + 1
                        self.depressive_words += 1
                    else:
                        self.tf_positive[word] = self.tf_positive.get(word, 0) + 1
                        self.positive_words += 1
                    if word not in count:
                        count += [word]
                for word in count:
                    if self.labels.iloc[i]:
                        self.idf_depressive[word] = self.idf_depressive.get(word, 0) + 1
                    else:
                        self.idf_positive[word] = self.idf_positive.get(word, 0) + 1

        def calc_TF_IDF(self):
            self.prob_depressive = dict()
            self.prob_positive = dict()
            self.sum_tf_idf_depressive = 0
            self.sum_tf_idf_positive = 0
            for word in self.tf_depressive:
                self.prob_depressive[word] = (self.tf_depressive[word]) * log(
                    (self.depressive_tweets + self.positive_tweets) \
                    / (self.idf_depressive[word] + self.idf_positive.get(word, 0)))
                self.sum_tf_idf_depressive += self.prob_depressive[word]
            for word in self.tf_depressive:
                self.prob_depressive[word] = (self.prob_depressive[word] + 1) / (
                            self.sum_tf_idf_depressive + len(list(self.prob_depressive.keys())))

            for word in self.tf_positive:
                self.prob_positive[word] = (self.tf_positive[word]) * log(
                    (self.depressive_tweets + self.positive_tweets) \
                    / (self.idf_depressive.get(word, 0) + self.idf_positive[word]))
                self.sum_tf_idf_positive += self.prob_positive[word]
            for word in self.tf_positive:
                self.prob_positive[word] = (self.prob_positive[word] + 1) / (
                            self.sum_tf_idf_positive + len(list(self.prob_positive.keys())))

            self.prob_depressive_tweet, self.prob_positive_tweet = self.depressive_tweets / self.total_tweets, self.positive_tweets / self.total_tweets

        def classify(self, processed_message):
            pDepressive, pPositive = 0, 0
            for word in processed_message:
                if word in self.prob_depressive:
                    pDepressive += log(self.prob_depressive[word])
                else:
                    if self.method == 'tf-idf':
                        pDepressive -= log(self.sum_tf_idf_depressive + len(list(self.prob_depressive.keys())))
                    else:
                        pDepressive -= log(self.depressive_words + len(list(self.prob_depressive.keys())))
                if word in self.prob_positive:
                    pPositive += log(self.prob_positive[word])
                else:
                    if self.method == 'tf-idf':
                        pPositive -= log(self.sum_tf_idf_positive + len(list(self.prob_positive.keys())))
                    else:
                        pPositive -= log(self.positive_words + len(list(self.prob_positive.keys())))
                pDepressive += log(self.prob_depressive_tweet)
                pPositive += log(self.prob_positive_tweet)
            return pDepressive >= pPositive

        def predict(self, testData):
            result = dict()
            for (i, message) in enumerate(testData):
                processed_message = process_message(message)
                result[i] = int(self.classify(processed_message))
            return result

    def metrics(labels, predictions):
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        for i in range(len(labels)):
            true_pos += int(labels.iloc[i] == 1 and predictions[i] == 1)
            true_neg += int(labels.iloc[i] == 0 and predictions[i] == 0)
            false_pos += int(labels.iloc[i] == 0 and predictions[i] == 1)
            false_neg += int(labels.iloc[i] == 1 and predictions[i] == 0)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        Fscore = 2 * precision * recall / (precision + recall)
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F-score: ", Fscore)
        print("Accuracy: ", accuracy)

    sc_tf_idf = TweetClassifier(trainData, 'tf-idf')
    sc_tf_idf.train()
    preds_tf_idf = sc_tf_idf.predict(testData['message'])
    metrics(testData['label'], preds_tf_idf)

    sc_bow = TweetClassifier(trainData, 'bow')
    sc_bow.train()
    preds_bow = sc_bow.predict(testData['message'])
    metrics(testData['label'], preds_bow)

    i = 0
    print(tweetsx)
    tweetsx.append('Hi hello depression and anxiety are the worst')
    for i in range(len(tweetsx)):
        pm = process_message(tweetsx[i])
        if sc_bow.classify(pm) == True:
            dep = "Depressed"
        else:
            dep = "Healthy"
    json_file = {}
    json_file['quer'] = dep
    return jsonify(json_file)


@app.route('/instasuiml/<id>')
def get_product(id):
    print(id)
    def preprocess_tweet(text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        text = re.sub('[\W]+', ' ', text.lower())
        text = text + ' '.join(emoticons).replace('-', '')
        return text

    tqdm.pandas()
    df = pd.read_csv('suicidal_data.csv')
    df['tweet'] = df['tweet'].progress_apply(preprocess_tweet)

    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()

    def tokenizer_porter(text):
        return [porter.stem(word) for word in text.split()]

    from nltk.corpus import stopwords
    stop = stopwords.words('english')

    [w for w in tokenizer_porter('a runner likes running and runs a lot') if w not in stop]

    def tokenizer(text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\(|D|P)', text.lower())
        text = re.sub('[\W]+', ' ', text.lower())
        text += ' '.join(emoticons).replace('-', '')
        tokenized = [w for w in tokenizer_porter(text) if w not in stop]
        return tokenized

    """### Using the Hashing Vectorizer"""

    from sklearn.feature_extraction.text import HashingVectorizer
    vect = HashingVectorizer(decode_error='ignore', n_features=2 ** 21,
                             preprocessor=None, tokenizer=tokenizer)

    """### Building the Model"""

    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss='log', random_state=1)

    X = df["tweet"].to_list()
    y = df['label']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=0)

    X_train = vect.transform(X_train)
    X_test = vect.transform(X_test)

    classes = np.array([0, 1])
    clf.partial_fit(X_train, y_train, classes=classes)

    print('Accuracy: %.3f' % clf.score(X_test, y_test))

    clf = clf.partial_fit(X_test, y_test)

    """### Testing and making Predictions"""

    # Commented out IPython magic to ensure Python compatibility.
    label = {0: 'negative', 1: 'positive'}
    example = ["I'll kill myself am tired of living depressed and alone"]
    X = vect.transform(example)
    print('Prediction: %s\nProbability: %.2f%%'
          % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X)) * 100))

    # Commented out IPython magic to ensure Python compatibility.
    label = {0: 'negative', 1: 'positive'}
    example = ["i will slit my wrist"]
    X = vect.transform(example)
    print('Prediction: %s\nProbability: %.2f%%'
          % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X)) * 100))

    conn = http.client.HTTPSConnection("simple-instagram-scraper1.p.rapidapi.com")

    headers = {
        'x-rapidapi-host': "simple-instagram-scraper1.p.rapidapi.com",
        'x-rapidapi-key': "e056db6af5msh5b87eb4727d8928p18cdddjsn6638d71b971c"
    }

    conn.request("GET", "/api/profile/"+id+"/posts", headers=headers)

    res = conn.getresponse()
    data = res.read()

    result = data.decode("utf-8")
    json_object = json.loads(result)
    json_formatted_str = json.dumps(json_object, indent=2)

    # print(json_formatted_str)
    json_object = json.dumps(json_formatted_str, indent=4)
    with open('readmedf.json', 'w') as f:
        f.write(json_formatted_str)

    f = open('readmedf.json')
    x = json.load(f)
    c = 0
    a = []
    for i in x:
        c = c + 1
        print(i['title'])
        a.append(i['title'])
    # print(c)
    print(a)
    tweets = []
    tweets = a
    f.close()

    pred = []
    for i in tweets:
        label = {0: 'negative', 1: 'positive'}
        a = []
        a.append(i)
        example = a
        X = vect.transform(example)
        pred.append(label[clf.predict(X)[0]])
        # print('Prediction: %s\nProbability: %.2f%%'
        # %(label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))
    # print(pred)
    if 'positive' in pred:
        b = "Suicidal"
        print('Suicidal tendencies detected')
    else:
        b = 'Healthy'

    json_file = {}
    json_file['quer'] = b
    return jsonify(json_file)


@app.route('/twittersuicideml/<id>')
def get_twittersuicideml(id):
    print(id)
    def preprocess_tweet(text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        text = re.sub('[\W]+', ' ', text.lower())
        text = text + ' '.join(emoticons).replace('-', '')
        return text

    tqdm.pandas()
    df = pd.read_csv('suicidal_data.csv')
    df['tweet'] = df['tweet'].progress_apply(preprocess_tweet)

    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()

    def tokenizer_porter(text):
        return [porter.stem(word) for word in text.split()]

    from nltk.corpus import stopwords
    stop = stopwords.words('english')

    [w for w in tokenizer_porter('a runner likes running and runs a lot') if w not in stop]

    def tokenizer(text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\(|D|P)', text.lower())
        text = re.sub('[\W]+', ' ', text.lower())
        text += ' '.join(emoticons).replace('-', '')
        tokenized = [w for w in tokenizer_porter(text) if w not in stop]
        return tokenized

    """### Using the Hashing Vectorizer"""

    from sklearn.feature_extraction.text import HashingVectorizer
    vect = HashingVectorizer(decode_error='ignore', n_features=2 ** 21,
                             preprocessor=None, tokenizer=tokenizer)

    """### Building the Model"""

    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss='log', random_state=1)

    X = df["tweet"].to_list()
    y = df['label']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=0)

    X_train = vect.transform(X_train)
    X_test = vect.transform(X_test)

    classes = np.array([0, 1])
    clf.partial_fit(X_train, y_train, classes=classes)

    print('Accuracy: %.3f' % clf.score(X_test, y_test))

    clf = clf.partial_fit(X_test, y_test)

    """### Testing and making Predictions"""

    # Commented out IPython magic to ensure Python compatibility.
    label = {0: 'negative', 1: 'positive'}
    example = ["I'll kill myself am tired of living depressed and alone"]
    X = vect.transform(example)
    print('Prediction: %s\nProbability: %.2f%%'
          % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X)) * 100))

    # Commented out IPython magic to ensure Python compatibility.
    # label = {0:'negative', 1:'positive'}
    # example = ["i will slit my wrist"]
    # X = vect.transform(example)
    # print('Prediction: %s\nProbability: %.2f%%'
    #       %(label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))

    url = 'https://twitter135.p.rapidapi.com/UserTweets/'

    querystring = {"id": id, "count": "1000"}

    headers = {
        'x-rapidapi-host': "twitter135.p.rapidapi.com",
        'x-rapidapi-key': "dd3b14fedamsh7fff9cbfe6800d9p19ff87jsn1ce1c277807c"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    x = response.text
    # print(x)

    import json
    y = json.loads(x)
    tweets = []
    # the result is a Python dictionary:
    for i in range(60):
        if "TimelineTimelineItem" in \
                y["data"]["user"]["result"]["timeline"]["timeline"]["instructions"][1]["entries"][i]["content"][
                    'entryType']:
            # print(y["data"]["user"]["result"]["timeline"]["timeline"]["instructions"][1]["entries"][i]["content"]['entryType'])
            tweets.append(
                y["data"]["user"]["result"]["timeline"]["timeline"]["instructions"][1]["entries"][i]["content"][
                    'itemContent']['tweet_results']['result']['legacy']['full_text'])
            # print(y["data"]["user"]["result"]["timeline_v2"]["timeline"]["instructions"][0]["entries"][i]["content"]["itemContent"]["tweet_results"]['result']['legacy']['full_text'])
    print(tweets)

    pred = []
    for i in tweets:
        label = {0: 'negative', 1: 'positive'}
        a = []
        a.append(i)
        example = a
        X = vect.transform(example)
        pred.append(label[clf.predict(X)[0]])
        # print('Prediction: %s\nProbability: %.2f%%'
        # %(label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))
    # print(pred)
    if 'positive' in pred:
        b = "Suicidal"
        print('Suicidal tendencies detected')
    else:
        b = 'Healthy'

    json_file = {}
    json_file['quer'] = b
    return jsonify(json_file)

@app.route('/twitterdepml/<id>')
def get_twitterdepml(id):
    print(id)
    url = "https://twitter135.p.rapidapi.com/UserTweets/"

    querystring = {"id": id, "count": "1000"}

    headers = {
        'x-rapidapi-host': "twitter135.p.rapidapi.com",
        'x-rapidapi-key': "dd3b14fedamsh7fff9cbfe6800d9p19ff87jsn1ce1c277807c"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    x = response.text
    # print(x)

    import json
    y = json.loads(x)
    tweetsx = []
    # the result is a Python dictionary:
    for i in range(60):
        if "TimelineTimelineItem" in \
                y["data"]["user"]["result"]["timeline"]["timeline"]["instructions"][1]["entries"][i]["content"][
                    'entryType']:
            # print(y["data"]["user"]["result"]["timeline"]["timeline"]["instructions"][1]["entries"][i]["content"]['entryType'])
            tweetsx.append(
                y["data"]["user"]["result"]["timeline"]["timeline"]["instructions"][1]["entries"][i]["content"][
                    'itemContent']['tweet_results']['result']['legacy']['full_text'])
            # print(y["data"]["user"]["result"]["timeline_v2"]["timeline"]["instructions"][0]["entries"][i]["content"]["itemContent"]["tweet_results"]['result']['legacy']['full_text'])
    # print(tweetsx)

    tweets = pd.read_csv('sentiment_tweets3.csv')
    tweets.head(20)

    tweets.drop(['Unnamed: 0'], axis=1, inplace=True)

    tweets['label'].value_counts()

    tweets.info()

    """# Splitting the Data in Training and Testing Sets

    As you can see, I used almost all the data for training: 98% and the rest for testing.
    """

    totalTweets = 8000 + 2314
    trainIndex, testIndex = list(), list()
    for i in range(tweets.shape[0]):
        if np.random.uniform(0, 1) < 0.98:
            trainIndex += [i]
        else:
            testIndex += [i]
    trainData = tweets.iloc[trainIndex]
    testData = tweets.iloc[testIndex]

    tweets.info()

    trainData['label'].value_counts()

    trainData.head()

    testData['label'].value_counts()

    testData.head()

    """# Wordcloud Analysis"""

    depressive_words = ' '.join(list(tweets[tweets['label'] == 1]['message']))

    positive_words = ' '.join(list(tweets[tweets['label'] == 0]['message']))

    """#Pre-processing the data for the training: Tokenization, stemming, and removal of stop words"""

    def process_message(message, lower_case=True, stem=True, stop_words=True, gram=2):
        if lower_case:
            message = message.lower()
        words = word_tokenize(message)
        words = [w for w in words if len(w) > 2]
        if gram > 1:
            w = []
            for i in range(len(words) - gram + 1):
                w += [' '.join(words[i:i + gram])]
            return w
        if stop_words:
            sw = stopwords.words('english')
            words = [word for word in words if word not in sw]
        if stem:
            stemmer = PorterStemmer()
            words = [stemmer.stem(word) for word in words]
        return words

    class TweetClassifier(object):
        def __init__(self, trainData, method='tf-idf'):
            self.tweets, self.labels = trainData['message'], trainData['label']
            self.method = method

        def train(self):
            self.calc_TF_and_IDF()
            if self.method == 'tf-idf':
                self.calc_TF_IDF()
            else:
                self.calc_prob()

        def calc_prob(self):
            self.prob_depressive = dict()
            self.prob_positive = dict()
            for word in self.tf_depressive:
                self.prob_depressive[word] = (self.tf_depressive[word] + 1) / (self.depressive_words + \
                                                                               len(list(self.tf_depressive.keys())))
            for word in self.tf_positive:
                self.prob_positive[word] = (self.tf_positive[word] + 1) / (self.positive_words + \
                                                                           len(list(self.tf_positive.keys())))
            self.prob_depressive_tweet, self.prob_positive_tweet = self.depressive_tweets / self.total_tweets, self.positive_tweets / self.total_tweets

        def calc_TF_and_IDF(self):
            noOfMessages = self.tweets.shape[0]
            self.depressive_tweets, self.positive_tweets = self.labels.value_counts()[1], self.labels.value_counts()[0]
            self.total_tweets = self.depressive_tweets + self.positive_tweets
            self.depressive_words = 0
            self.positive_words = 0
            self.tf_depressive = dict()
            self.tf_positive = dict()
            self.idf_depressive = dict()
            self.idf_positive = dict()
            for i in range(noOfMessages):
                message_processed = process_message(self.tweets.iloc[i])
                count = list()  # To keep track of whether the word has ocured in the message or not.
                # For IDF
                for word in message_processed:
                    if self.labels.iloc[i]:
                        self.tf_depressive[word] = self.tf_depressive.get(word, 0) + 1
                        self.depressive_words += 1
                    else:
                        self.tf_positive[word] = self.tf_positive.get(word, 0) + 1
                        self.positive_words += 1
                    if word not in count:
                        count += [word]
                for word in count:
                    if self.labels.iloc[i]:
                        self.idf_depressive[word] = self.idf_depressive.get(word, 0) + 1
                    else:
                        self.idf_positive[word] = self.idf_positive.get(word, 0) + 1

        def calc_TF_IDF(self):
            self.prob_depressive = dict()
            self.prob_positive = dict()
            self.sum_tf_idf_depressive = 0
            self.sum_tf_idf_positive = 0
            for word in self.tf_depressive:
                self.prob_depressive[word] = (self.tf_depressive[word]) * log(
                    (self.depressive_tweets + self.positive_tweets) \
                    / (self.idf_depressive[word] + self.idf_positive.get(word, 0)))
                self.sum_tf_idf_depressive += self.prob_depressive[word]
            for word in self.tf_depressive:
                self.prob_depressive[word] = (self.prob_depressive[word] + 1) / (
                            self.sum_tf_idf_depressive + len(list(self.prob_depressive.keys())))

            for word in self.tf_positive:
                self.prob_positive[word] = (self.tf_positive[word]) * log(
                    (self.depressive_tweets + self.positive_tweets) \
                    / (self.idf_depressive.get(word, 0) + self.idf_positive[word]))
                self.sum_tf_idf_positive += self.prob_positive[word]
            for word in self.tf_positive:
                self.prob_positive[word] = (self.prob_positive[word] + 1) / (
                            self.sum_tf_idf_positive + len(list(self.prob_positive.keys())))

            self.prob_depressive_tweet, self.prob_positive_tweet = self.depressive_tweets / self.total_tweets, self.positive_tweets / self.total_tweets

        def classify(self, processed_message):
            pDepressive, pPositive = 0, 0
            for word in processed_message:
                if word in self.prob_depressive:
                    pDepressive += log(self.prob_depressive[word])
                else:
                    if self.method == 'tf-idf':
                        pDepressive -= log(self.sum_tf_idf_depressive + len(list(self.prob_depressive.keys())))
                    else:
                        pDepressive -= log(self.depressive_words + len(list(self.prob_depressive.keys())))
                if word in self.prob_positive:
                    pPositive += log(self.prob_positive[word])
                else:
                    if self.method == 'tf-idf':
                        pPositive -= log(self.sum_tf_idf_positive + len(list(self.prob_positive.keys())))
                    else:
                        pPositive -= log(self.positive_words + len(list(self.prob_positive.keys())))
                pDepressive += log(self.prob_depressive_tweet)
                pPositive += log(self.prob_positive_tweet)
            return pDepressive >= pPositive

        def predict(self, testData):
            result = dict()
            for (i, message) in enumerate(testData):
                processed_message = process_message(message)
                result[i] = int(self.classify(processed_message))
            return result

    def metrics(labels, predictions):
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        for i in range(len(labels)):
            true_pos += int(labels.iloc[i] == 1 and predictions[i] == 1)
            true_neg += int(labels.iloc[i] == 0 and predictions[i] == 0)
            false_pos += int(labels.iloc[i] == 0 and predictions[i] == 1)
            false_neg += int(labels.iloc[i] == 1 and predictions[i] == 0)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        Fscore = 2 * precision * recall / (precision + recall)
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F-score: ", Fscore)
        print("Accuracy: ", accuracy)

    sc_tf_idf = TweetClassifier(trainData, 'tf-idf')
    sc_tf_idf.train()
    preds_tf_idf = sc_tf_idf.predict(testData['message'])
    metrics(testData['label'], preds_tf_idf)

    sc_bow = TweetClassifier(trainData, 'bow')
    sc_bow.train()
    preds_bow = sc_bow.predict(testData['message'])
    metrics(testData['label'], preds_bow)
    dep = ""
    i = 0
    for i in range(len(tweetsx)):
        pm = process_message(tweetsx[i])
        if sc_bow.classify(pm) == True:
            dep = "Depressed"
        else:
            dep = "Healthy"
    json_file = {}
    json_file['quer'] = dep
    return jsonify(json_file)


if __name__ == '__main__':
    app.run()
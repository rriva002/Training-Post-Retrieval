import csv
import re
from json import loads
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


data_dir = "data/"


def construct_datasets(filename, topic, test_length, test_pos_ratio, pos, neg,
                       use_stemmer, random_state):
    stem = SnowballStemmer("english") if use_stemmer else None
    related, non_related = [], []

    with open(filename, "r") as file:
        for row in csv.reader(file):
            if row[1].lower() == topic.lower():
                related.append(clean_str(row[0], stem))
            else:
                non_related.append(clean_str(row[0], stem))

    test_size_pos = round(test_length * test_pos_ratio)
    test_size_neg = test_length - test_size_pos
    pos_train, pos_test = train_test_split(related, test_size=test_size_pos,
                                           random_state=random_state)
    neg_train, neg_test = train_test_split(non_related,
                                           test_size=test_size_neg,
                                           random_state=random_state)
    X_train = pos_train + neg_train
    X_test = pos_test + neg_test
    y_train = [pos] * len(pos_train) + [neg] * len(neg_train)
    y_test = [pos] * len(pos_test) + [neg] * len(neg_test)
    return X_train, y_train, X_test, y_test


def clean_str(string, stemmer=None):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()

    if stemmer is not None:
        string = " ".join([stemmer.stem(w) for w in word_tokenize(string)])

    return string


def load_data(source, topic, test_length, test_pos_ratio=0.5, pos="Positive",
              neg="Negative", use_stemmer=False, random_state=None):
    if source == "ds":
        return load_ds_data(topic, test_length, test_pos_ratio, pos, neg,
                            use_stemmer, random_state=random_state)
    elif source == "huffpost":
        return load_huffpost_data(topic, test_length, test_pos_ratio, pos, neg,
                                  use_stemmer, random_state=random_state)
    elif source == "reddit":
        return load_reddit_data(topic, test_length, test_pos_ratio, pos, neg,
                                use_stemmer, random_state=random_state)


def load_ds_data(topic, test_length, test_pos_ratio=0.5, pos="Positive",
                 neg="Negative", use_stemmer=False, random_state=None):
    filename = data_dir + "ds_data.csv"
    return construct_datasets(filename, topic, test_length, test_pos_ratio,
                              pos, neg, use_stemmer, random_state)


def load_huffpost_data(category, test_length, test_pos_ratio=0.5,
                       pos="Positive", neg="Negative", use_stemmer=False,
                       random_state=None):
    stem = SnowballStemmer("english") if use_stemmer else None
    related, non_related = [], []
    mappings = {"ARTS": "ARTS & CULTURE", "CULTURE & ARTS": "ARTS & CULTURE",
                "PARENTS": "PARENTING", "STYLE": "STYLE & BEAUTY",
                "THE WORLDPOST": "WORLDPOST", "WELLNESS": "HEALTHY LIVING"}

    with open("data/News_Category_Dataset_v2.json", "r") as file:
        for line in file:
            json = loads(line)
            text = " ".join([json["headline"], json["short_description"]])
            extracted_category = json["category"]

            if extracted_category in mappings:
                extracted_category = mappings[extracted_category]

            if extracted_category == category:
                related.append(clean_str(text, stem))
            else:
                non_related.append(clean_str(text, stem))

    test_size_pos = round(test_length * test_pos_ratio)
    test_size_neg = test_length - test_size_pos
    pos_train, pos_test = train_test_split(related, test_size=test_size_pos,
                                           random_state=random_state)
    neg_train, neg_test = train_test_split(non_related,
                                           test_size=test_size_neg,
                                           random_state=random_state)
    X_train = pos_train + neg_train
    X_test = pos_test + neg_test
    y_train = [pos] * len(pos_train) + [neg] * len(neg_train)
    y_test = [pos] * len(pos_test) + [neg] * len(neg_test)
    return X_train, y_train, X_test, y_test


def load_reddit_data(subreddit, test_length, test_pos_ratio=0.5,
                     pos="Positive", neg="Negative", use_stemmer=False,
                     random_state=None):
    filename = data_dir + "reddit_data.csv"
    return construct_datasets(filename, subreddit, test_length, test_pos_ratio,
                              pos, neg, use_stemmer, random_state)

import csv
import re
from json import loads
from random import sample, seed


data_dir = "data/"


def assign_fold_ids(X, y, num_folds, random_state):
    seed(random_state)

    indices = sample(range(len(X)), len(X))
    fold_size = int(len(X) / num_folds)
    fold_ids = []

    for fold_id in range(num_folds):
        fold_ids += [fold_id for _ in range(fold_size)]

    fold_ids += [fold_id for fold_id in range(len(X) - len(fold_ids))]
    return [X[i] for i in indices], [y[i] for i in indices], fold_ids


def construct_cv_dataset(filename, num_folds, random_state):
    X, y = [], []

    with open(filename, "r", encoding="utf-8") as file:
        for row in csv.reader(file):
            X.append(clean_str(row[0]))
            y.append(row[1].lower())

    return assign_fold_ids(X, y, num_folds, random_state)


def clean_str(string):
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
    return string.strip().lower()


def load_data(source, num_cv_folds=5, pos="Positive", neg="Negative",
              random_state=None):
    if source == "ds":
        return load_ds_data(num_cv_folds, pos, neg, random_state=random_state)
    elif source == "huffpost":
        return load_huffpost_data(num_cv_folds, pos, neg,
                                  random_state=random_state)
    elif source == "reddit":
        return load_reddit_data(num_cv_folds, pos, neg,
                                random_state=random_state)


def load_ds_data(num_cv_folds=5, pos="Positive", neg="Negative",
                 random_state=None):
    filename = data_dir + "ds_data.csv"
    return construct_cv_dataset(filename, num_cv_folds, random_state)


def load_huffpost_data(num_cv_folds=5, pos="Positive", neg="Negative",
                       random_state=None):
    mappings = {"ARTS": "ARTS & CULTURE", "CULTURE & ARTS": "ARTS & CULTURE",
                "PARENTS": "PARENTING", "STYLE": "STYLE & BEAUTY",
                "THE WORLDPOST": "WORLDPOST", "WELLNESS": "HEALTHY LIVING"}
    X, y = [], []

    with open(data_dir + "News_Category_Dataset_v2.json", "r") as file:
        for line in file:
            json = loads(line)
            text = " ".join([json["headline"], json["short_description"]])
            extracted_category = json["category"]

            if extracted_category in mappings:
                extracted_category = mappings[extracted_category]

            X.append(clean_str(text))
            y.append(extracted_category.lower())

    return assign_fold_ids(X, y, num_cv_folds, random_state)


def load_reddit_data(num_cv_folds=5, pos="Positive", neg="Negative",
                     random_state=None):
    filename = data_dir + "reddit_data.csv"
    return construct_cv_dataset(filename, num_cv_folds, random_state)

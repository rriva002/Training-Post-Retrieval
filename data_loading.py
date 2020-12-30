import csv
import re
from json import loads
from os import makedirs, path
from pathlib import Path
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

    if not path.exists(data_dir):
        makedirs(data_dir)

    if not Path(filename).is_file():
        source = "http://www.dailystrength.org"
        sql = "SELECT url, body, disorder FROM {} WHERE source = '{}' "
        sql += "AND url LIKE '{}/group/%' AND replyid = 0 AND body IS NOT NULL"
        urls = set()
        connection = mysql_connection()
        cursor = connection.cursor()

        with open(filename, "w") as file:
            writer = csv.writer(file, lineterminator="\n")

            cursor.execute(sql.format("healthforumposts", source, source))

            for (url, body, disorder) in cursor:
                text = str(body)

                if url in urls or len(text.strip()) == 0:
                    continue

                urls.add(url)
                writer.writerow([text.replace("\n", " "), disorder])

        del urls

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

    if not path.exists(data_dir):
        makedirs(data_dir)

    if not Path(filename).is_file():
        ecig, source = "Electronic Cigarette", "https://www.reddit.com"
        sql = "SELECT url, body, disorder FROM {} WHERE source = '{}' "
        sql += "AND replyid = 0 AND body != '[removed]' AND body IS NOT NULL"
        urls = set()
        connection = mysql_connection()
        cursor = connection.cursor()

        with open(filename, "w") as file:
            writer = csv.writer(file, lineterminator="\n")

            for table in ["healthforumposts", "ecigarette"]:
                cursor.execute(sql.format(table, source))

                for (url, body, disorder) in cursor:
                    text = str(body)

                    if url in urls or len(text.strip()) == 0:
                        continue

                    sub = ecig if table == "ecigarette" else disorder

                    urls.add(url)
                    writer.writerow([text.replace("\n", " "), sub])

        del urls

    return construct_cv_dataset(filename, num_cv_folds, random_state)


def mysql_connection():
    from mysql import connector
    return connector.connect(host="dblab-rack20", database="HEALTHDATA",
                             user="rriva002", password="passwd", use_pure=True)

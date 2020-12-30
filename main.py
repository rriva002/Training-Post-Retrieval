import csv
from cnn_text_classification import CNNClassifier
from data_loading import load_data
from json import loads
from math import exp
from os import makedirs, path
from pathlib import Path
from queue import Queue
from random import sample, seed
from scipy.stats import entropy
from simulated_api import SimulatedAPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from text_dataset_labeling_assistant import TextDatasetLabelingAssistant as LA
from threading import Lock, Thread
from time import time

default_data_lengths = [100 * (i + 1) for i in range(10)]
default_ksas = ["all_keywords", "50_50", "tp_ksa", "tpp_ksa", "tprn_ksa",
                "tpprn_ksa", "liu", "random", "ideal"]
default_stats = ["accuracy", "precision", "recall", "percent_positive",
                 "kl_pos", "kl_neg", "balance_and_diversity", "time"]


class CVThread(Thread):
    def __init__(self, assist, classifier, data_lengths, generate_keywords,
                 keywords, ksas, lock, n_keywords, neg, pos, queue,
                 random_state, results, stats, topic, *args, **kwargs):
        self.assist = assist
        self.classifier = classifier
        self.data_lengths = data_lengths
        self.queue = queue
        self.generate_kw = generate_keywords
        self.kw = keywords
        self.ksas = ksas
        self.lock = lock
        self.n_keywords = n_keywords
        self.neg = neg
        self.pos = pos
        self.random_state = random_state
        self.results = results
        self.stats = stats
        self.topic = topic

        super().__init__(*args, **kwargs)

    def run(self):
        while not self.queue.empty():
            fold, num_cv_folds = self.queue.get()

            seed(self.random_state)
            self.assist.get_api().set_test_fold_id(fold)

            if "kl_pos" in self.stats or "kl_neg" in self.stats:
                random_samples = generate_random_samples(self.assist.get_api(),
                                                         self.pos, self.neg,
                                                         self.data_lengths[-1])
            else:
                random_samples = None

            if self.generate_kw or self.kw is None or len(self.kw) == 0:
                directory = "keywords/{}".format(source)
                t = self.topic.lower().replace(" ", "_")
                fn = "{}/{}_keywords_{}_{}.txt".format(directory, t, fold,
                                                       num_cv_folds)

                if not path.exists(directory):
                    makedirs(directory)

                if Path(fn).is_file():
                    with open(fn, "r") as file:
                        keywords = [line.rstrip("\n") for line in file]
                        keywords = keywords[:min(len(keywords),
                                                 self.n_keywords)]
                else:
                    keywords = ig_keywords(self.assist.get_api(),
                                           self.n_keywords, self.pos, fn,
                                           random_state=self.random_state)
            else:
                keywords = self.kw

            for ksa in self.ksas:
                for m in self.data_lengths:
                    print("    Fold {}: {}, m={}".format(fold, ksa, m))
                    test(self.topic, keywords, self.assist, self.classifier,
                         m, self.pos, self.neg, self.results, random_samples,
                         ksa=ksa, random_state=self.random_state,
                         lock=self.lock)

            self.queue.task_done()


def batch_experiments(source, classifier="cnn", generate_keywords=True,
                      n_keywords=5, num_cv_folds=5, num_threads="cv",
                      random_state=None, ksas=default_ksas,
                      stats=default_stats, data_lengths=default_data_lengths):
    pos, neg = "Positive", "Negative"
    topics, keywords = [], []

    print("Source: " + source)

    with open("{}.json".format(source), "r") as file:
        for topic, manual_keywords in loads(file.read()).items():
            topics.append(topic)
            keywords.append(manual_keywords)

    X, y, cv_fold_ids = load_data(source, num_cv_folds=num_cv_folds, pos=pos,
                                  neg=neg, random_state=random_state)
    assist = LA(SimulatedAPI(X, labels=y, cv_fold_ids=cv_fold_ids))
    percentage_stats = {"accuracy", "balanced_accuracy", "percent_positive"}
    results = dict(zip(topics, [{} for topic in topics]))
    queue, lock = Queue(), Lock()

    for topic in topics:
        for ksa in ksas:
            results[topic][ksa] = dict(zip(stats, [{} for stat in stats]))

            for stat in stats:
                for m in data_lengths:
                    results[topic][ksa][stat][m] = []

    for i, topic in enumerate(topics):
        print("  Topic: " + topic)
        assist.get_api().assign_binary_labels(topic, pos=pos, neg=neg)

        for fold in range(num_cv_folds):
            queue.put((fold, num_cv_folds))

        for _ in range(num_cv_folds if num_threads == "cv" else num_threads):
            CVThread(assist, classifier, data_lengths, generate_keywords,
                     keywords[i], ksas, lock, n_keywords, neg, pos, queue,
                     random_state, results, stats, topic).start()

        queue.join()

        for ksa in ksas:
            for stat in stats:
                for m in data_lengths:
                    result = sum(results[topic][ksa][stat][m]) / num_cv_folds

                    if stat in percentage_stats:
                        result = "%0.2f" % (100 * result) + "%"
                    else:
                        result = "%0.4f" % result

                    results[topic][ksa][stat][m] = result

    with open("{}_results.csv".format(source), "w") as file:
        writer = csv.writer(file, lineterminator="\n")

        for topic in topics:
            header = [topic]

            for stat in stats:
                header += [stat] + ["" for _ in data_lengths]

            writer.writerow(header)
            writer.writerow([""] + ["m={}".format(m) for m in data_lengths])

            for ksa in ksas:
                row = [ksa]

                for stat in stats:
                    r = results[topic][ksa][stat]
                    row += [r[m] for m in data_lengths] + [""]

                writer.writerow(row[:-1])

            writer.writerow([""])


def generate_random_samples(api, pos, neg, m):
    X, y = api.train_data()
    Xp = [X[i] for i in range(len(X)) if y[i] == pos]
    Xn = [X[i] for i in range(len(X)) if y[i] == neg]
    return {pos: sample(Xp, min(len(Xp), m)), neg: sample(Xn, min(len(Xn), m))}


def ig_keywords(api, n, positive, filename, min_df=2, max_df=1.0,
                random_state=None):
    X, y = api.train_data()
    vec = CountVectorizer(min_df=min_df, max_df=max_df, stop_words="english")

    vec.fit([X[i] for i in range(len(X)) if y[i] == positive])

    ig = mutual_info_classif(vec.transform(X), y, random_state=random_state)
    non_keywords = {"quot", "039", "http", "https", "com", "www"}
    keywords = [(word, ig[i]) for word, i in vec.vocabulary_.items()]
    keywords = sorted(keywords, key=lambda keyword: keyword[1], reverse=True)
    keywords = [kw[0] for kw in keywords if kw[0] not in non_keywords]

    with open(filename, "w") as file:
        file.writelines([keyword + "\n" for keyword in keywords])

    return keywords[:min(n, len(keywords))]


def kl(X, y, label, random_sample, min_df=1, max_df=1.0):
    vec = CountVectorizer(min_df=min_df, max_df=max_df)
    X_l = [X[i] for i in range(len(X)) if y[i] == label]

    try:
        q = vec.fit_transform(X_l).sum(axis=0, dtype="float")
        p = vec.transform(random_sample[:len(X_l)]).sum(axis=0, dtype="float")
    except ValueError:
        return float("inf")

    return entropy(p, q, axis=1)[0]


def select_classifier(classifier, random_state=None, active_learning=False):
    if classifier == "cnn":
        scoring = make_scorer(balanced_accuracy_score)
        return CNNClassifier(scoring=scoring, vectors="fasttext.en.300d",
                             random_state=random_state, static=True, epochs=50,
                             batch_size=100)

    vectorizer = TfidfVectorizer(max_features=1000, min_df=0.03,
                                 ngram_range=(1, 3), stop_words="english")

    if classifier == "rf":
        classifier = RandomForestClassifier(n_estimators=2000, n_jobs=-1,
                                            random_state=random_state,
                                            class_weight="balanced")
    elif classifier == "svm":
        classifier = SVC(kernel="linear", probability=active_learning,
                         class_weight="balanced", random_state=random_state)

    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])


def test(topic, keywords, assist, classifier, m, positive, negative, results,
         random_samples, ksa="50_50", random_state=None, lock=None):
    assist.get_api().set_random_state(random_state)

    stats = results[topic][ksa].keys()
    lock_early = classifier == "cnn"
    start = time()

    if ksa == "50_50":
        X, y = assist.fifty_fifty(keywords, m)
    elif ksa == "tp_ksa":
        X, y = assist.tp_ksa(keywords, m)
    elif ksa == "tpp_ksa":
        X, y = assist.tp_ksa(keywords, m, proportional=True)
    elif ksa == "tprn_ksa":
        X, y = assist.tp_ksa(keywords, m, rand_neg=True)
    elif ksa == "tpprn_ksa":
        X, y = assist.tp_ksa(keywords, m, proportional=True, rand_neg=True)
    elif ksa == "all_keywords":
        X, y = assist.all_keywords(keywords, m)
    elif ksa == "active_learning":
        if lock is not None and lock_early:
            lock.acquire()

        classifier = select_classifier(classifier, random_state=random_state,
                                       active_learning=True)
        X, y = assist.active_learning(classifier, m, keywords=keywords,
                                      batch_size=int(m / 10))
    elif ksa == "liu":
        X, y = assist.liu(keywords, m)
    elif ksa == "random":
        X, y = assist.random(m)
    elif ksa == "ideal":
        X, y = assist.ideal(m)

    elapsed = round(time() - start)
    X_test, y_test = assist.get_api().test_data()

    if lock is not None and ksa != "active_learning" and lock_early:
        lock.acquire()

    if len(set(y)) == 1:
        predictions = [y[0] for x in X_test]
    else:
        if ksa != "active_learning":
            classifier = select_classifier(classifier,
                                           random_state=random_state)
            classifier.fit(X, y)

        predictions = classifier.predict(X_test)

    if lock is not None and not lock_early:
        lock.acquire()

    for stat in stats:
        if stat == "accuracy":
            value = accuracy_score(y_test, predictions)
        elif stat == "balanced_accuracy":
            value = balanced_accuracy_score(y_test, predictions)
        elif stat == "precision":
            value = precision_score(y_test, predictions, pos_label=positive,
                                    zero_division=0)
        elif stat == "recall":
            value = recall_score(y_test, predictions, pos_label=positive)
        elif stat == "percent_positive":
            n_pos = len([X[i] for i in range(len(X)) if y[i] == positive])
            value = n_pos / len(X)
        elif stat == "kl_pos":
            value = kl(X, y, positive, random_samples[positive])
        elif stat == "kl_neg":
            value = kl(X, y, negative, random_samples[negative])
        elif stat == "time":
            value = elapsed

        results[topic][ksa][stat][m].append(value)

    if "balance_and_diversity" in stats and "percent_positive" in stats and \
            "kl_pos" in stats and "kl_neg" in stats:
        percent_positive = results[topic][ksa]["percent_positive"][m][-1]
        balance = 1 - 2 * abs(0.5 - percent_positive)
        diversity_pos = exp(-results[topic][ksa]["kl_pos"][m][-1])
        diversity_neg = exp(-results[topic][ksa]["kl_neg"][m][-1])

        if percent_positive > 0 and diversity_pos > 0 and diversity_neg > 0:
            bd = 3 / (1 / balance + 1 / diversity_pos + 1 / diversity_neg)
            results[topic][ksa]["balance_and_diversity"][m].append(bd)
        else:
            results[topic][ksa]["balance_and_diversity"][m].append(0)

    if lock is not None:
        lock.release()


if __name__ == "__main__":
    for source in ["ds", "huffpost", "reddit"]:
        batch_experiments(source, random_state=10)

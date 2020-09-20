import csv
from data_loading import load_data
from json import loads
from numpy import transpose
from os import makedirs, path
from pathlib import Path
from random import sample, seed
from scipy.stats import entropy
from simulated_api import SimulatedAPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from text_dataset_labeling_assistant import TextDatasetLabelingAssistant as LA
from cnn_text_classification import CNNClassifier
from time import time

default_data_lengths = [100 * i for i in range(1, 11)]
default_modes = ["50_50_naive", "single_keyword", "top_k", "top_k_prop",
                 "top_k_randneg", "top_k_prop_randneg", "liu"]
default_stats = ["Accuracy", "Percent Positive", "Diversity"]


def batch_experiments(source, experiment_type="keyword", method="cnn",
                      random_state=10, generate_keywords=True, n_keywords=5,
                      modes=default_modes, stats=default_stats,
                      data_lengths=default_data_lengths):
    pos, neg, rs = "Positive", "Negative", random_state
    topics, keywords, results = [], [], {}

    with open("{}.json".format(source), "r") as file:
        for topic, manual_keywords in loads(file.read()).items():
            topics.append(topic)
            keywords.append(manual_keywords)

    keywords_old = keywords

    for i in range(len(topics)):
        seed(random_state)
        print(topics[i])

        results[topics[i]] = {}
        classifier = configure_classifier(method, random_state=random_state)
        X_train, y_train, X_test, y_test = load_data(source, topics[i], 1000,
                                                     pos=pos, neg=neg,
                                                     random_state=rs)

        if experiment_type == "keyword":
            assist = LA(SimulatedAPI(X_train, y_train))

        if "Diversity" in stats:
            Xp = [X_train[i] for i in range(len(X_train)) if y_train[i] == pos]
            Xn = [X_train[i] for i in range(len(X_train)) if y_train[i] == neg]
            random_pos = sample(range(len(Xp)), min(len(Xp), data_lengths[-1]))
            random_neg = sample(range(len(Xn)), min(len(Xn), data_lengths[-1]))
            random_samples = {pos: [Xp[i] for i in random_pos],
                              neg: [Xn[i] for i in random_neg]}
            Xp, Xn, random_pos, random_neg = None, None, None, None
        else:
            random_samples = None

        if experiment_type == "keyword":
            if generate_keywords or keywords is None or len(keywords) == 0:
                directory = "keywords/{}".format(source)
                t = topics[i].lower().replace(" ", "_")
                filename = "{}/{}_{}_keywords.txt".format(directory, source, t)

                if not path.exists(directory):
                    makedirs(directory)

                keywords = select_keywords(X_train, y_train, n_keywords, pos,
                                           filename, random_state=rs)
            else:
                keywords = keywords[i]
        elif experiment_type == "random":
            print(experiment_type + ":")

            r = test_random_labeling(X_train, y_train, data_lengths,
                                     classifier, X_test, y_test, pos, stats,
                                     random_samples)

            results[topics[i]][experiment_type] = r
            continue
        elif experiment_type == "ideal":
            print(experiment_type + ":")

            r = test_ideal_labeling(X_train, y_train, data_lengths, classifier,
                                    X_test, y_test, pos, stats, random_samples,
                                    random_state=rs, remove_first_kw=False)
            results[topics[i]][experiment_type] = r
            continue

        print(" ".join(["Keywords:"] + keywords))

        for mode in modes:
            r = test_keyword_labeling(keywords, data_lengths, assist,
                                      classifier, X_test, y_test, pos, stats,
                                      random_samples, mode=mode)
            results[topics[i]][mode] = r

        del X_train
        del y_train
        del assist

        keywords = keywords_old

    filename = "{}_results_{}.csv".format(source, experiment_type)

    with open(filename, "w") as file:
        writer = csv.writer(file, lineterminator="\n")
        modes = [experiment_type] if experiment_type != "keyword" else modes
        stats = sorted(list(results[topics[0]][modes[0]].keys()))

        for topic in topics:
            header = [topic]

            for stat in stats:
                header += [stat] + ["" for ln in data_lengths]

            writer.writerow(header)
            writer.writerow([""] + ["m={}".format(ln) for ln in data_lengths])

            for mode in modes:
                if mode not in results[topic]:
                    continue

                row = [mode]

                for stat in stats:
                    r = results[topic][mode][stat]
                    row += [r[ln] for ln in data_lengths] + [""]

                writer.writerow(row[:-1])

            writer.writerow([""])


def build_results_dict(stats):
    results = {}

    for stat in stats:
        if stat == "Percent with Keyword" or stat == "Diversity":
            results[stat + " (Positives)"] = {}
            results[stat + " (Negatives)"] = {}
        else:
            results[stat] = {}

    return results


def configure_classifier(classifier, grid_search=False, random_state=None):
    scoring = make_scorer(balanced_accuracy_score)
    n_jobs = 1 if classifier == "cnn" or classifier == "rf" else -1

    if classifier == "cnn":
        classifier = CNNClassifier(random_state=random_state, scoring=scoring,
                                   vectors="fasttext.en.300d")
        param_grid = {"kernel_sizes": [(2, 3, 4), (3, 4, 5), (4, 5, 6)],
                      "kernel_num": [100, 200, 300, 400, 500, 600]}
    else:
        vectorizer = TfidfVectorizer(max_features=1000, min_df=0.03,
                                     ngram_range=(1, 3), stop_words="english")
        param_grid = {"vectorizer__max_features": [500, 1000, None],
                      "vectorizer__min_df": [1, 0.01, 0.03]}

        if classifier == "rf":
            classifier = RandomForestClassifier(n_estimators=2000, n_jobs=-1,
                                                random_state=random_state,
                                                class_weight="balanced")
            param_grid["classifier__n_estimators"] = [10, 100, 1000]
            param_grid["classifier__max_depth"] = [2, 10, None]
        elif classifier == "svm":
            classifier = LinearSVC(random_state=random_state, max_iter=10000,
                                   class_weight="balanced")
            param_grid["classifier__C"] = [0.001, 0.01, 0.1, 1.0, 10.0]
            param_grid["classifier__loss"] = ["hinge", "squared_hinge"]

        classifier = Pipeline([("vectorizer", vectorizer),
                               ("classifier", classifier)])

    if grid_search:
        return GridSearchCV(classifier, scoring=scoring, cv=5, iid=False,
                            param_grid=param_grid, n_jobs=n_jobs)

    return classifier


def kl(X, y, label, random_sample, min_df=1, max_df=1.0):
    vec = CountVectorizer(min_df=min_df, max_df=max_df)
    X_l = [X[i] for i in range(len(X)) if y[i] == label]

    try:
        q = vec.fit_transform(X_l).sum(axis=0, dtype="float")
        p = vec.transform(random_sample[:len(X_l)]).sum(axis=0, dtype="float")
    except ValueError:
        return float("inf")

    return entropy(transpose(p), transpose(q))


def prepare_results(results, stats, test_results, elapsed, data_len):
    stats = set(stats)
    accuracy = "Accuracy"
    bal_acc = "Balanced Accuracy"
    pct_pos = "Percent Positive"
    pct_kw = "Percent with Keyword"
    precision = "Precision"
    recall = "Recall"
    diversity = "Diversity"
    time = "Time"
    fmt_2, fmt_4 = "%0.2f", "%0.4f"

    if accuracy in stats:
        results[accuracy][data_len] = fmt_2 % test_results["accuracy"] + "%"

    if bal_acc in stats:
        results[bal_acc][data_len] = fmt_2 % test_results["balanced"] + "%"

    if pct_pos in stats:
        results[pct_pos][data_len] = fmt_2 % test_results["positive"] + "%"

    if pct_kw in stats:
        kw_pos, kw_neg = pct_kw + " (Positives)", pct_kw + " (Negatives)"
        results[kw_pos][data_len] = fmt_2 % test_results["kw_pos"] + "%"
        results[kw_neg][data_len] = fmt_2 % test_results["kw_neg"] + "%"

    if precision in stats:
        results[precision][data_len] = fmt_4 % test_results["precision"]

    if recall in stats:
        results[recall][data_len] = fmt_4 % test_results["recall"]

    if diversity in stats:
        div_p, div_n = diversity + " (Positives)", diversity + " (Negatives)"
        results[div_p][data_len] = fmt_4 % test_results["diversity_positive"]
        results[div_n][data_len] = fmt_4 % test_results["diversity_negative"]

    if time in stats:
        results[time][data_len] = str(round(elapsed))

    return results


def print_results(data_lengths, results):
    for stat in sorted(list(results.keys())):
        print(stat + ":\n" + "\t".join([str(ln) for ln in data_lengths]))
        print("\t".join([results[stat][ln] for ln in data_lengths]))
        print("")


def select_keywords(X, y, n, positive, filename, min_df=2, max_df=1.0,
                    random_state=None):
    if Path(filename).is_file():
        with open(filename, "r") as file:
            keywords = [line.rstrip("\n") for line in file]
            return keywords[:min(len(keywords), n)]

    vec = CountVectorizer(min_df=min_df, max_df=max_df, stop_words="english")

    vec.fit([X[i] for i in range(len(X)) if y[i] == positive])

    ig = mutual_info_classif(vec.transform(X), y, random_state=random_state)
    non_keywords = set(["quot", "039", "http", "https", "com", "www"])
    keywords = [(word, ig[i]) for word, i in vec.vocabulary_.items()]
    keywords = sorted(keywords, key=lambda keyword: keyword[1], reverse=True)
    keywords = [kw[0] for kw in keywords if kw[0] not in non_keywords]

    with open(filename, "w") as file:
        file.writelines([keyword + "\n" for keyword in keywords])

    return keywords[:min(n, len(keywords))]


def test(classifier, X, y, test_data, test_labels, positive, random_state,
         keyword=None):
    if isinstance(classifier, GridSearchCV):
        classifier.estimator.set_params(classifier__random_state=random_state)
    elif isinstance(classifier, Pipeline):
        classifier.set_params(classifier__random_state=random_state)
    else:
        classifier.set_params(random_state=random_state)

    if len(set(y)) == 1:
        predictions = [y[0] for x in test_data]
    else:
        classifier.fit(X, y)
        predictions = classifier.predict(test_data)

    acc = 100 * accuracy_score(test_labels, predictions)
    bal_acc = 100 * balanced_accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, pos_label=positive)
    recall = recall_score(test_labels, predictions, pos_label=positive)
    positives = [X[i] for i in range(len(X)) if y[i] == positive]
    pct_pos = 100 * len(positives) / len(X)
    results = {"accuracy": acc, "balanced": bal_acc, "positive": pct_pos,
               "precision": precision, "recall": recall}

    if keyword is not None:
        negatives = [X[i] for i in range(len(X)) if y[i] != positive]
        kw_pos = sum([1 for x in positives if keyword in x]) / len(positives)
        kw_neg = sum([1 for x in negatives if keyword in x]) / len(negatives)
        results["kw_pos"] = 100 * kw_pos
        results["kw_neg"] = 100 * kw_neg

    return results


def test_ideal_labeling(X_train, y_train, data_lengths, classifier, X_test,
                        y_test, positive, stats, random_samples, filename=None,
                        random_state=10, remove_first_kw=False):
    results = build_results_dict(stats)
    X_pos = [X_train[i] for i in range(len(X_train)) if y_train[i] == positive]
    X_neg = [X_train[i] for i in range(len(X_train)) if y_train[i] != positive]
    y_neg = [y_train[i] for i in range(len(y_train)) if y_train[i] != positive]

    if filename is not None and Path(filename).is_file():
        with open(filename, "r") as file:
            keyword = [line.rstrip("\n") for line in file][0]
    else:
        keyword = None

    for data_length in data_lengths:
        seed(random_state)

        start = time()
        half = int(round(data_length / 2))
        pos_indices = sample(range(len(X_pos)), half)
        neg_indices = sample(range(len(X_neg)), data_length - half)
        X = [X_pos[index] for index in pos_indices]

        if remove_first_kw and keyword is not None:
            X = [x for x in X if keyword not in x]

        y = [positive for i in range(len(X))]
        X += [X_neg[index] for index in neg_indices]
        y += [y_neg[index] for index in neg_indices]
        elapsed = round(time() - start)
        rs = test(classifier, X, y, X_test, y_test, positive,
                  random_state, keyword=keyword)

        if "Diversity" in stats:
            for label, random_sample in random_samples.items():
                key = "diversity_{}".format(label.lower())
                rs[key] = kl(X, y, label, random_samples[label])

        results = prepare_results(results, stats, rs, elapsed, data_length)

    print_results(data_lengths, results)
    return results


def test_keyword_labeling(keywords, data_lengths, assist, classifier,
                          test_data, test_labels, positive, stats,
                          random_samples, mode="50_50", use_all_kw=False,
                          random_state=10, filename=None):
    results = build_results_dict(stats)

    for data_length in data_lengths:
        assist.get_api().set_random_state(random_state)

        start = time()

        if mode == "50_50_naive":
            X, y = assist.label_with_naive_50_50(keywords[0], data_length)
        elif mode == "top_k":
            X, y = assist.label_with_top_k(keywords, data_length,
                                           rand_neg=False)
        elif mode == "top_k_prop":
            X, y = assist.label_with_top_k(keywords, data_length,
                                           proportional=True, rand_neg=False)
        elif mode == "top_k_randneg":
            X, y = assist.label_with_top_k(keywords, data_length,
                                           rand_neg=True)
        elif mode == "top_k_prop_randneg":
            X, y = assist.label_with_top_k(keywords, data_length,
                                           proportional=True, rand_neg=True)
        elif mode == "single_keyword":
            X, y = assist.get_api().query([keywords[0]], data_length)

            if len(X) < data_length:
                X_r, y_r = assist.get_api().query(None, data_length - len(X))
                X += X_r
                y += y_r
        elif mode == "liu":
            X, y = assist.label_with_liu(keywords, data_length)

        assist.get_api().reset()

        elapsed = round(time() - start)
        rs = test(classifier, X, y, test_data, test_labels, positive,
                  random_state)

        if "Diversity" in stats:
            for label, random_sample in random_samples.items():
                key = "diversity_{}".format(label.lower())
                rs[key] = kl(X, y, label, random_samples[label])

        results = prepare_results(results, stats, rs, elapsed, data_length)

        if filename is not None:
            with open(filename, "w") as file:
                writer = csv.writer(file, lineterminator="\n")

                for i in range(len(X)):
                    writer.writerow([X[i], y[i]])

    print(mode + ":")
    print_results(data_lengths, results)
    return results


def test_random_labeling(data, labels, data_lengths, classifier, test_data,
                         test_labels, positive, stats, random_samples,
                         random_state=10):
    results = build_results_dict(stats)

    for data_length in data_lengths:
        seed(random_state)

        start = time()
        indices = sample(range(len(data)), data_length)
        X = [data[index] for index in indices]
        y = [labels[index] for index in indices]
        elapsed = round(time() - start)
        rs = test(classifier, X, y, test_data, test_labels, positive,
                  random_state)

        if "Diversity" in stats:
            for label, random_sample in random_samples.items():
                key = "diversity_{}".format(label.lower())
                rs[key] = kl(X, y, label, random_samples[label])

        results = prepare_results(results, stats, rs, elapsed, data_length)

    print_results(data_lengths, results)
    return results

batch_experiments("ds", "keyword")
batch_experiments("ds", "ideal")
batch_experiments("ds", "random")
batch_experiments("reddit", "keyword")
batch_experiments("reddit", "ideal")
batch_experiments("reddit", "random")

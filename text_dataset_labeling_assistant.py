from keyword_ranking import KeywordRanking
from numpy import argmax, argsort, array
from queue import Queue
from random import sample
from scipy.stats import entropy


default_labels = ["Positive", "Negative"]


class TextDatasetLabelingAssistant(object):
    def __init__(self, api, sampling_ratio=0.2, no_kw_throw_exception=False):
        self.__api = api
        self.__sampling_ratio = sampling_ratio
        self.__no_kw_throw_exception = no_kw_throw_exception

    def __dist(self, x, y, a, b, c):
        return (a * x + b * y + c) / (a ** 2 + b ** 2) ** 0.5

    def __distances_from_line(self, kw):
        a = kw[-1][1] - kw[0][1]
        b = -(len(kw) - 1)
        c = (len(kw) - 1) * kw[0][1]
        return [self.__dist(i, kw[i][1], a, b, c) for i in range(len(kw))]

    def __labeling_prompt(self, document, doc_num, m, valid_input, labels):
        if document[1] is not None:
            return document[1]

        label = None

        print("Document {} of {}:".format(doc_num, m))
        print(document[0] + "\n\nSelect a class label:")

        for j in range(len(labels)):
            print(valid_input[j] + ": " + labels[j])

        while label is None or label not in valid_input:
            selection = input("\n> ")

            if selection in valid_input:
                label = labels[int(selection) - 1]

        return label

    def __query_top_k(self, keywords, queried, m, pc_pos, method=""):
        X, y = [], []

        if queried < m:
            budgets = self.__top_k_budgets(keywords, pc_pos, m - queried,
                                           method)

            for i in range(len(keywords) - 1, -1, -1):
                if queried + budgets[i] > m:
                    budgets[i] = max(0, m - queried)

                X_k, y_k = self.__api.query([keywords[i]], budgets[i])
                X += X_k
                y += y_k
                queried += len(X_k)

                if len(X_k) < budgets[i]:
                    if self.__no_kw_throw_exception:
                        msg = "Keyword {}: Budget {}, returned {}"
                        msg = msg.format(keywords[i], budgets[i], len(X_k))
                        raise NameError(msg)

                    budgets = self.__top_k_budgets(keywords[:i], pc_pos[:i],
                                                   m - queried, method)

            if queried < m:
                X_q, y_q = self.__api.query(None, m - queried)
                X += X_q
                y += y_q

        return X, y

    def __sample_50_50(self, keyword, m, valid_labels=default_labels):
        num_samples = int(self.__sampling_ratio * m)
        X, y = self.__api.query([keyword], max(30, num_samples))
        X, y = self.label(X, y, m, valid_labels, [], [])
        pos_count = sum([1 for label in y if label == valid_labels[0]])
        pos_ratio = pos_count / len(y) if len(y) > 0 else 0
        return X, y, 1.0 if pos_ratio < 0.5 else 0.5 / pos_ratio

    def __select_top_k(self, keywords, m, valid_labels,
                       use_all_keywords=False):
        positive = valid_labels[0]
        tot_sample = m * self.__sampling_ratio
        sample_sz = max(30, int(tot_sample / len(keywords)))
        X, y, pct_pos = [], [], []

        for i, keyword in enumerate(keywords):
            if sample_sz + len(X) > 0.8 * m:
                break

            X_k, y_k = self.__api.query([keyword], sample_sz, keywords[:i])
            X_k, y_k = self.label(X_k, y_k, m, valid_labels, [], [])
            pos_count = sum([1 for label in y_k if label == positive])
            percent_positive = 0 if len(y_k) == 0 else pos_count / len(y_k)

            pct_pos.append(percent_positive)

            X += X_k
            y += y_k

        if use_all_keywords:
            sel_kw = keywords
        elif len(pct_pos) > 2:
            kw = [(keywords[i], pct_pos[i]) for i in range(len(pct_pos))]
            kw = sorted(kw, key=lambda keyword: keyword[1], reverse=True)
            dist = self.__distances_from_line(kw)
            max_arg = argmax(dist)
            kw = kw[:len(kw) - 1 if max_arg == 0 else max_arg]
            sel_kw = [keyword[0] for keyword in kw]
            pct_pos = [keyword[1] for keyword in kw]
        else:
            sel_kw = [keywords[0]]
            pct_pos = [pct_pos[0]]

        # fmt = "{}: {}%"
        # fmt_pp = ["%0.2f" % (100 * p) for p in pct_pos]
        # fmt_kw = [fmt.format(sel_kw[i], p) for i, p in enumerate(fmt_pp)]

        # print(" ".join(fmt_kw), end="")
        return X, y, sel_kw, pct_pos

    def __top_k_budgets(self, keywords, pc_pos, total_budget, method):
        if len(keywords) == 0:
            return []
        elif method == "prop":
            sum_of_percentages = sum(pc_pos)

            if sum_of_percentages > 0:
                x = total_budget / sum_of_percentages
                return [int(x * percentage) for percentage in pc_pos]
        elif method == "zipf":
            ratios = [1 / i for i in range(1, len(keywords) + 1)]
            x = total_budget / sum(ratios)
            return [int(x * ratio) for ratio in ratios]

        return [int(total_budget / len(keywords))] * len(keywords)

    def get_api(self):
        return self.__api

    def active_learning(self, classifier, m, keywords=None, pool_size=100000,
                        batch_size=1, prob_batch_size=64,
                        valid_labels=default_labels):
        use_keywords = keywords is not None and len(keywords) > 0
        labels_seen = set()
        X, y = [], []

        if use_keywords:
            q = Queue(maxsize=len(keywords) + 1)

            for keyword in keywords + [None]:
                q.put(keyword)

        while len(labels_seen) < len(valid_labels) and len(X) < m:
            if use_keywords:
                keyword = q.get()
                X_q, y_q = self.__api.query(keyword, 1)

                q.put(keyword)
            else:
                X_q, y_q = self.__api.query(None, 1)

            if len(X_q) > 0:
                X, y = self.label(X_q, y_q, m, valid_labels, X=X, y=y)

                labels_seen.add(y[-1])

        if len(labels_seen) == 1:
            return X, y

        pool = set(self.__api.get_active_learning_pool_indices(pool_size))

        classifier.fit(X, y)

        while len(X) < m:
            batch_size = min(batch_size, m - len(X), len(pool))
            indices = list(pool)
            probs = []
            prob_batch_start = 0

            if prob_batch_size == "all":
                prob_batch_size = len(pool)

            while prob_batch_start < len(pool):
                prob_batch_end = prob_batch_start + prob_batch_size
                prob_batch_end = min(prob_batch_end, len(pool))
                prob_batch_indices = indices[prob_batch_start:prob_batch_end]

                X_p, _ = self.__api.get_data(prob_batch_indices)
                probs += list(classifier.predict_proba(X_p))
                prob_batch_start = prob_batch_end

            top_indices = argsort(-1 * entropy(array(probs), base=2, axis=1))
            top_indices = [indices[i] for i in top_indices[:batch_size]]
            X_t, y_t = self.__api.get_data(top_indices)
            X, y = self.label(X_t, y_t, m, valid_labels, X=X, y=y)

            classifier.fit(X, y)
            pool.difference_update(top_indices)

            prob_batch_size = min(prob_batch_size, len(pool))

        return X, y

    def all_keywords(self, keywords, m, valid_labels=default_labels):
        budget = int(m / len(keywords))
        X, y = [], []

        for keyword in keywords:
            X_k, y_k = self.__api.query([keyword], budget)
            X += X_k
            y += y_k

        if len(X) < m:
            X_r, y_r = self.__api.query(None, m - len(X))
            X += X_r
            y += y_r

        return self.label(X, y, m, valid_labels, [], [])

    def fifty_fifty(self, keywords, m, valid_labels=default_labels):
        budget = int(m / (2 * len(keywords)))
        X, y = [], []

        for keyword in keywords:
            X_k, y_k = self.__api.query([keyword], budget)
            X += X_k
            y += y_k

        X_r, y_r = self.__api.query(None, m - len(X))
        X += X_r
        y += y_r
        return self.label(X, y, m, valid_labels, [], [])

    def ideal(self, m, valid_labels=default_labels):
        X, y = [], []
        X_train, y_train = self.get_api().train_data()
        budget = int(m / len(valid_labels))
        indices = dict([(label, []) for label in valid_labels])

        for i, label in enumerate(y_train):
            indices[label].append(i)

        for label in valid_labels:
            for i in sample(indices[label], budget):
                X.append(X_train[i])
                y.append(label)

        return X, y

    def label(self, data, labels, m, valid_labels=default_labels, X=[], y=[]):
        valid_input = [str(x + 1) for x in range(len(valid_labels))]

        for i in range(len(data)):
            label = None if labels is None else labels[i]
            label = self.__labeling_prompt((data[i], label), len(X) + 1, m,
                                           valid_input, valid_labels)

            X.append(data[i])
            y.append(label)

        return X, y

    def liu(self, keywords, m, num_iterations=2, valid_labels=default_labels):
        keyword_ranking = KeywordRanking(minimum_frequency=0.05)
        X, y, _, pc_ps = self.__select_top_k(keywords, m,
                                             valid_labels=valid_labels)
        current_docs = [X[i] for i in range(len(X)) if y[i] == valid_labels[0]]
        ref_docs = [X[i] for i in range(len(X)) if y[i] != valid_labels[0]]

        for i in range(num_iterations):
            keywords = keyword_ranking.double_ranking(current_docs, ref_docs,
                                                      keywords, self.__api)

        Xq, yq = self.__query_top_k(keywords, len(X), m, pc_ps)
        X, y = self.label(Xq, yq, m, valid_labels, X=X, y=y)

        if len(X) < m:
            X_r, y_r = self.__api.query(None, m - len(X))
            X, y = self.label(X_r, y_r, m, valid_labels=valid_labels, X=X, y=y)

        return X, y

    def random(self, m, valid_labels=default_labels):
        X, y = self.__api.query(None, m)
        return self.label(X, y, m, valid_labels, [], [])

    def set_sampling_ratio(self, sampling_ratio):
        self.__sampling_ratio = sampling_ratio

    def tp_ksa(self, keywords, m, proportional=False, rand_neg=False,
               valid_labels=default_labels):
        X, y, kw, pct_pos = self.__select_top_k(keywords, m, valid_labels)
        m_k = m / (1 + sum(pct_pos) / len(pct_pos)) if rand_neg else m
        m_k = round(min(m_k, 0.8 * m)) if rand_neg else m
        method = "prop" if proportional else ""
        Xq, yq = self.__query_top_k(kw, len(X), m_k, pct_pos, method=method)
        X, y = self.label(Xq, yq, m, valid_labels, X, y)
        num_queried = len(X)

        if rand_neg:
            X = [X[i] for i in range(len(X)) if y[i] == valid_labels[0]]
            y = [valid_labels[0] for x in X]

        X_r, y_r = self.__api.query(None, m - num_queried)
        return self.label(X_r, y_r, m, valid_labels, X, y)

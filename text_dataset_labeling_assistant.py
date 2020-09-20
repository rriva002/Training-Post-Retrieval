from keyword_ranking import KeywordRanking
from numpy import argmax, transpose
from scipy.stats import entropy


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

    def get_api(self):
        return self.__api

    def label(self, data, labels, num_docs_to_label,
              valid_labels=["Positive", "Negative"], X=[], y=[]):
        valid_input = [str(x + 1) for x in range(len(valid_labels))]

        for i in range(len(data)):
            label = None if labels is None else labels[i]
            label = self.__labeling_prompt((data[i], label), str(len(X) + 1),
                                           num_docs_to_label, valid_input,
                                           valid_labels)

            X.append(data[i])
            y.append(label)

        return X, y

    def label_with_active_learning(self, data, classifier, vectorizer,
                                   num_docs_to_label, batch_size=1,
                                   valid_labels=["Positive", "Negative"],
                                   X=[], y=[]):
        valid_input = [str(x + 1) for x in range(len(valid_labels))]
        i = 0

        while len(y) < len(valid_labels) and len(data) > 0:
            label = self.__labeling_prompt(data[i], str(len(y) + 1),
                                           len(valid_labels), valid_input,
                                           valid_labels)

            if label not in y:
                X.append(data[i][0])
                y.append(label)
                del data[i]
            else:
                i += 1

        num_docs_labeled = len(y)

        while len(data) > 0 and num_docs_labeled < num_docs_to_label:
            if vectorizer is None:
                classifier.fit(X, y)

                probs = classifier.predict_proba(data)
            else:
                classifier.fit(vectorizer.fit_transform(X), y)
                vectors = vectorizer.transform([doc[0] for doc in data])
                probs = classifier.predict_proba(vectors)

            r = range(probs.shape[0])
            scores = [entropy(transpose(probs[i, :]), base=2) for i in r]

            del probs

            mappings = dict(zip([doc[0] for doc in data], scores))

            del scores
            data.sort(key=lambda doc: mappings[doc[0]], reverse=True)
            del mappings

            batch_size = min(batch_size, num_docs_to_label - num_docs_labeled,
                             len(data))

            for i in range(batch_size):
                num_docs_labeled += 1

                X.append(data[i][0])
                y.append(self.__labeling_prompt(data[i], str(num_docs_labeled),
                                                num_docs_to_label, valid_input,
                                                valid_labels))

            del data[:batch_size]

        return X, y

    def label_with_50_50(self, keyword, num_docs_to_label,
                         valid_labels=["Positive", "Negative"]):
        X, y, alpha = self.__sample_50_50(keyword, num_docs_to_label,
                                          valid_labels=valid_labels)
        x = int(alpha * num_docs_to_label) - len(X)
        X_q, y_q = self.__api.query([keyword], max(x, 0))
        X, y = self.label(X_q, y_q, num_docs_to_label, valid_labels, X=X, y=y)
        X_q, y_q = self.__api.query(None, max(num_docs_to_label - len(X), 0))
        X, y = self.label(X_q, y_q, num_docs_to_label, valid_labels, X=X, y=y)
        return X, y

    def label_with_liu(self, keywords, num_docs_to_label, num_iterations=2,
                       valid_labels=["Positive", "Negative"]):
        keyword_ranking = KeywordRanking(minimum_frequency=0.05)
        X, y, _, pc_ps = self.__select_top_k(keywords, num_docs_to_label,
                                             valid_labels=valid_labels)
        current_docs = [X[i] for i in range(len(X)) if y[i] == valid_labels[0]]
        ref_docs = [X[i] for i in range(len(X)) if y[i] != valid_labels[0]]

        for i in range(num_iterations):
            keywords = keyword_ranking.double_ranking(current_docs, ref_docs,
                                                      keywords, self.__api)

        Xq, yq = self.__query_top_k(keywords, len(X), num_docs_to_label, pc_ps)
        X, y = self.label(Xq, yq, num_docs_to_label, valid_labels, X=X, y=y)

        if len(X) < num_docs_to_label:
            X_r, y_r = self.__api.query(None, num_docs_to_label - len(X))
            X, y = self.label(X_r, y_r, num_docs_to_label,
                              valid_labels=valid_labels, X=X, y=y)

        return X, y

    def label_with_naive_50_50(self, keyword, num_docs_to_label,
                               valid_labels=["Positive", "Negative"]):
        half = int(num_docs_to_label / 2)
        X, y = self.__api.query([keyword], half)
        X, y = self.label(X, y, num_docs_to_label, valid_labels, [], [])
        X_r, y_r = self.__api.query(None, num_docs_to_label - half)
        return self.label(X_r, y_r, num_docs_to_label, valid_labels, X=X, y=y)

    def label_with_top_k(self, keywords, m, proportional=False, rand_neg=False,
                         valid_labels=["Positive", "Negative"]):
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

    def __labeling_prompt(self, document, doc_num, num_docs_to_label,
                          valid_input, labels):
        if document[1] is not None:
            return document[1]

        label = None

        print("Document " + doc_num + " of " + str(num_docs_to_label) + ":")
        print(document[0] + "\n\nSelect a class label:")

        for j in range(len(labels)):
            print(valid_input[j] + ": " + labels[j])

        while label is None or label not in valid_input:
            selection = input("\n> ")

            if selection in valid_input:
                label = labels[int(selection) - 1]

        return label

    def __query_top_k(self, keywords, queried, num_docs_to_label, pc_pos,
                      method="", alpha_weight=False):
        X = []
        y = []

        if queried < num_docs_to_label:
            budgets = self.__top_k_budgets(keywords, pc_pos,
                                           num_docs_to_label - queried, method)

            for i in range(len(keywords) - 1, -1, -1):
                if alpha_weight:
                    alpha = 1.0 if pc_pos[i] < 0.5 else 0.5 / pc_pos[i]
                    budgets[i] = int(alpha * budgets[i])

                if queried + budgets[i] > num_docs_to_label:
                    budgets[i] = max(0, num_docs_to_label - queried)

                X_k, y_k = self.__api.query([keywords[i]], budgets[i])
                X += X_k
                y += y_k
                queried += len(X_k)
                msg = "Keyword {}: Budget {}, returned {}"
                msg = msg.format(keywords[i], budgets[i], len(X_k))

                if len(X_k) < budgets[i]:
                    if self.__no_kw_throw_exception:
                        raise NameError(msg)

                    budgets = self.__top_k_budgets(keywords[:i], pc_pos[:i],
                                                   num_docs_to_label - queried,
                                                   method)

            if queried < num_docs_to_label:
                X_q, y_q = self.__api.query(None, num_docs_to_label - queried)
                X += X_q
                y += y_q

        return X, y

    def __sample_50_50(self, keyword, num_docs_to_label,
                       valid_labels=["Positive", "Negative"]):
        num_samples = int(self.__sampling_ratio * num_docs_to_label)
        X, y = self.__api.query([keyword], max(30, num_samples))
        X, y = self.label(X, y, num_docs_to_label, valid_labels, [], [])
        pos_count = sum([1 for label in y if label == valid_labels[0]])
        pos_ratio = pos_count / len(y) if len(y) > 0 else 0
        return X, y, 1.0 if pos_ratio < 0.5 else 0.5 / pos_ratio

    def __select_top_k(self, keywords, num_docs_to_label, valid_labels,
                       use_all_keywords=False):
        positive = valid_labels[0]
        tot_sample = num_docs_to_label * self.__sampling_ratio
        sample_sz = max(30, int(tot_sample / len(keywords)))
        X, y, pct_pos = [], [], []

        for i in range(len(keywords)):
            if sample_sz + len(X) > 0.8 * num_docs_to_label:
                break

            X_k, y_k = self.__api.query([keywords[i]], sample_sz, keywords[:i])
            X_k, y_k = self.label(X_k, y_k, num_docs_to_label, valid_labels,
                                  [], [])
            pos_count = sum([1 for label in y_k if label == positive])
            percent_positive = 0 if len(y_k) == 0 else pos_count / len(y_k)

            # pct_pos.append(1 - 2 * abs(percent_positive - 0.5))
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

        k = len(pct_pos)
        fmt = "{}: {}"
        fmt_kw = [fmt.format(sel_kw[i], 100 * pct_pos[i]) for i in range(k)]

        print(" ".join(fmt_kw))
        return X, y, sel_kw, pct_pos

    def set_sampling_ratio(self, sampling_ratio):
        self.__sampling_ratio = sampling_ratio

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

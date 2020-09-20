from nltk.tokenize import word_tokenize
from random import seed, sample


class SimulatedAPI(object):
    def __init__(self, data, labels=None, random_data=None, random_labels=None,
                 remove_queried=False):
        self.__data = data
        self.__indices_added = set()
        self.__labels = labels
        self.__query_cache = {}
        self.__remove_queried = remove_queried
        self.__tokenized = [set(word_tokenize(document)) for document in data]

        if random_data is not None:
            self.__random_api = SimulatedAPI(random_data, random_labels,
                                             remove_queried=remove_queried)
        else:
            self.__random_api = None

    def get_remove_queried(self):
        return self.__remove_queried

    def query(self, keywords, n=1, negated_keywords=[]):
        if keywords is None or len(keywords) == 0:
            if self.__random_api is not None:
                return self.__random_api.query(None, n)

            sample_range = set(range(len(self.__data)))
            indices = sample(sample_range.difference(self.__indices_added), n)
            X = [self.__data[i] for i in indices]

            if self.__labels is not None:
                y = [self.__labels[i] for i in indices]
            else:
                y = [None] * len(X)

            if self.__remove_queried:
                self.__indices_added.update(indices)

            return X, y

        key = " ".join(keywords + ["-" + k for k in negated_keywords])

        if key not in self.__query_cache:
            self.__query_cache[key] = set()
            keywords = set(keywords)
            negated_keywords = set(negated_keywords)

            for i in range(len(self.__tokenized)):
                t = self.__tokenized[i]

                if len(keywords.intersection(t)) > 0 and \
                        len(negated_keywords.intersection(t)) == 0:
                    self.__query_cache[key].add(i)

        sample_range = set(self.__query_cache[key])

        if self.__remove_queried:
            sample_range = sample_range.difference(self.__indices_added)

        if len(sample_range) > 0:
            indices = sample(sample_range, min(n, len(sample_range)))
            X = [self.__data[i] for i in indices]

            if self.__labels is not None:
                y = [self.__labels[i] for i in indices]
            else:
                y = [None] * len(X)

            if self.__remove_queried:
                self.__indices_added.update(indices)

            return X, y

        return [], []

    def reset(self):
        self.__indices_added.clear()

        if self.__random_api is not None:
            self.__random_api.reset()

    def set_random_state(self, random_state):
        seed(random_state)

        if self.__random_api is not None:
            self.__random_api.set_random_state(random_state)

    def set_remove_queried(self, remove_queried):
        self.__remove_queried = remove_queried

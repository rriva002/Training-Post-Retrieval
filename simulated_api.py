from nltk.tokenize import word_tokenize
from random import seed, sample
from threading import get_ident, Lock


class SimulatedAPI(object):
    def __init__(self, data, labels=None, cv_fold_ids=None):
        self.__cv_fold_ids = cv_fold_ids
        self.__data = data
        self.__labels = None
        self.__lock = Lock()
        self.__query_cache = {}
        self.__query_indices = {}
        self.__tokenized = [set(word_tokenize(document)) for document in data]
        self.__topics = labels

    def __len__(self):
        return len(self.__data)

    def assign_binary_labels(self, topic, pos="Positive", neg="Negative"):
        if self.__topics is not None:
            t1 = topic.lower()
            self.__labels = [pos if t1 == t2 else neg for t2 in self.__topics]

    def get_active_learning_pool_indices(self, pool_size):
        query_indices = self.__query_indices[get_ident()]
        indices = [i for i in range(len(self.__data)) if i in query_indices]
        return indices if pool_size == "all" else sample(indices, pool_size)

    def get_data(self, indices):
        X = [self.__data[index] for index in indices]

        if self.__labels is None:
            return X, [None] * len(X)

        y = [self.__labels[index] for index in indices]
        return X, y

    def query(self, keywords, n=1, negated_keywords=[]):
        query_indices = self.__query_indices[get_ident()]

        if keywords is None or len(keywords) == 0:
            indices = sample(query_indices, n)
            X = [self.__data[i] for i in indices]

            if self.__labels is not None:
                y = [self.__labels[i] for i in indices]
            else:
                y = [None] * len(X)

            return X, y

        sample_range = set()
        keywords = [keywords] if isinstance(keywords, str) else keywords
        negated_keywords = [negated_keywords] if \
            isinstance(negated_keywords, str) else negated_keywords

        for keyword in keywords:
            self.update_cache(keyword)
            sample_range.update(self.__query_cache[keyword])

        for keyword in negated_keywords:
            self.update_cache(keyword)
            sample_range.difference_update(self.__query_cache[keyword])

        sample_range.intersection_update(query_indices)

        if len(sample_range) > 0:
            indices = sample(sample_range, min(n, len(sample_range)))
            X = [self.__data[i] for i in indices]

            if self.__labels is not None:
                y = [self.__labels[i] for i in indices]
            else:
                y = [None] * len(X)

            return X, y

        return [], []

    def set_random_state(self, random_state):
        seed(random_state)

    def set_test_fold_id(self, test_fold_id):
        thread_id = get_ident()

        self.__lock.acquire()

        if self.__cv_fold_ids is not None:
            self.__query_indices[thread_id] = set()

            for i in range(len(self.__data)):
                if self.__cv_fold_ids[i] != test_fold_id:
                    self.__query_indices[thread_id].add(i)

        self.__lock.release()

    def test_data(self):
        query_indices = self.__query_indices[get_ident()]
        indices = range(len(self.__data))
        X = [self.__data[i] for i in indices if i not in query_indices]
        y = [self.__labels[i] for i in indices if i not in query_indices]
        return X, y

    def train_data(self):
        query_indices = self.__query_indices[get_ident()]
        indices = range(len(self.__data))
        X = [self.__data[i] for i in indices if i in query_indices]
        y = [self.__labels[i] for i in indices if i in query_indices]
        return X, y

    def update_cache(self, keyword):
        self.__lock.acquire()

        if keyword not in self.__query_cache:
            self.__query_cache[keyword] = set()

            for i, tokens in enumerate(self.__tokenized):
                if keyword in tokens:
                    self.__query_cache[keyword].add(i)

        self.__lock.release()

from gensim.parsing.preprocessing import STOPWORDS
from math import log
from nltk.tokenize import word_tokenize
from operator import itemgetter


class KeywordRanking(object):
    def __init__(self, minimum_frequency=0.01):
        self.__minimum_frequency = minimum_frequency

    def __entropy(self, f_current_docs, f_ref_docs, lmb=1.0):
        f_s = [f_current_docs, f_ref_docs]
        len_s = len(f_s)
        denom = sum([f_s[i] + lmb for i in range(len(f_s))])
        return -sum([self.__e_term(f_s, lmb, i, denom) for i in range(len_s)])

    def __e_term(self, f_s, lmb, i, denom):
        return (f_s[i] + lmb) / denom * log((f_s[i] + lmb) / denom, 2.0)

    def __frequency(self, word, documents):
        frequency = 0

        for document in documents:
            frequency += 1 if word in document else 0

        return frequency

    def __word_frequencies(self, documents):
        word_frequencies = {}

        for document in documents:
            for word in word_tokenize(document):
                if word not in word_frequencies:
                    word_frequencies[word] = 0

                word_frequencies[word] += 1

        return word_frequencies

    def __initial_ranking(self, current_docs, ref_docs, current_keywords,
                          unsuitable_words):
        f_current_docs = self.__word_frequencies(current_docs)
        f_ref_docs = self.__word_frequencies(ref_docs)
        word_entropy = {}

        for word in f_current_docs.keys():
            if word not in current_keywords and word not in unsuitable_words \
                    and "'" not in word and "\\" not in word \
                    and "," not in word:
                f_rd_word = f_ref_docs[word] if word in f_ref_docs else 0
                frequency = f_current_docs[word] + f_rd_word
                frequency /= len(current_docs) + len(ref_docs)

                if frequency > self.__minimum_frequency:
                    if f_current_docs[word] > f_rd_word:
                        e = self.__entropy(f_current_docs[word], f_rd_word)
                        word_entropy[word] = e

        shortlist = list(word_entropy.items())
        shortlist = sorted(shortlist, key=lambda keyword: keyword[1])
        shortlist = [keyword[0] for keyword in shortlist]
        return shortlist[:min(100, len(shortlist))]

    def __re_ranking(self, candidate_words, keywords, api):
        candidates_relevance_value = {}

        for one_candidate in candidate_words:
            matched_result = {}
            one_candidate_word_returned_result = {}
            relevance = -1.0
            cwl = one_candidate.lower()
            matched_result[cwl] = 0
            one_candidate_word_returned_result[cwl] = 0
            valid = 0
            lines, _ = api.query([cwl], 300)

            for line in lines:
                clean_line = line.strip()
                tokens_one_tweet = clean_line.split(" ")
                tokens_one_tweet_lower = []

                for one_token_raw in tokens_one_tweet:
                    tokens_one_tweet_lower.append(one_token_raw.lower())

                if clean_line != "":
                    valid += 1
                    one_candidate_word_returned_result[cwl] += 1
                    matched = False

                    for seed in keywords:
                        if seed.lower() in tokens_one_tweet_lower:
                            matched = True

                    if matched:
                        matched_result[cwl] += 1

            for key in matched_result:
                if one_candidate_word_returned_result[key] != 0:
                    relevance = float(matched_result[key]) / float(valid)

            candidates_relevance_value[cwl] = relevance

        sorted_candidates = list(candidates_relevance_value.items())
        sorted_candidates.sort(key=itemgetter(1), reverse=True)
        sorted_candidates = [candidate[0] for candidate in sorted_candidates]
        return sorted_candidates[:min(len(sorted_candidates), len(keywords))]

    def double_ranking(self, current_docs, ref_docs, keywords, api,
                       unsuitable_words=STOPWORDS):
        shortlist = self.__initial_ranking(current_docs, ref_docs, keywords,
                                           unsuitable_words)
        return self.__re_ranking(shortlist, keywords, api)

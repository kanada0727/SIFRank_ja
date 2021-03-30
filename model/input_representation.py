#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19

import nltk
from stanza import Pipeline
import MeCab
import os
from app.config import Config

with open(
    os.path.join(
        os.path.dirname(__file__),
        "../" "/auxiliary_data/japanese_stopwords.txt",
    )
) as f:
    stop_words = [line.rstrip("\n") for line in f]
    stopword_dict = set(stop_words)


class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, ja_model: Pipeline, text=""):
        """
        :param is_sectioned: If we want to section the text.
        :param ja_model: the pipeline of tokenization and POS-tagger
        :param considered_tags: The POSs we want to keep
        """
        self.phrase_extractor = PhraseExtractor()

        # get Document object of stanza (latest StanfordCoreNLP model)
        doc = ja_model(text)
        self.tokens = [
            word.text for sentence in doc.sentences for word in sentence.words
        ]
        self.tokens_tagged = [
            (word.text, word.xpos)
            for sentence in doc.sentences
            for word in sentence.words
        ]

        # validate tokens and tagged tokens contain the same, then override stopwords tags
        assert len(self.tokens) == len(self.tokens_tagged)
        for i, token in enumerate(self.tokens):
            if token.lower() in stopword_dict:
                self.tokens_tagged[i] = (token, "IN")

        self.keyphrase_candidate = self.phrase_extractor.extract_candidates(text)


class PhraseExtractor:
    def __init__(self):

        # GRAMMAR1 is the general way to extract NPs
        self.GRAMMAR1 = """  NP:
                {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

        self.GRAMMAR2 = """  NP:
                {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""

        self.GRAMMAR3 = """  NP:
                {<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

    def extract_candidates(self, text):
        """
        Based on part of speech return a list of candidate phrases
        :param text_obj: Input text Representation see @InputTextObj
        :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
        """
        """
        np_parser = nltk.RegexpParser(self.GRAMMAR1)  # Noun phrase parser
        keyphrase_candidate = []
        np_pos_tag_tokens = np_parser.parse(tokens_tagged)
        count = 0
        for token in np_pos_tag_tokens:
            if isinstance(token, nltk.tree.Tree) and token._label == "NP":
                np = "".join(word for word, tag in token.leaves())
                length = len(token.leaves())
                start_end = (count, count + length)
                count += length
                keyphrase_candidate.append((np, start_end))

            else:
                count += 1

        return keyphrase_candidate
        """
        return extract_keyphrase_candidates(text)


def tokenize(text):
    wakati = MeCab.Tagger(
        f"-r /dev/null -d {Config.mecab.system_dic.ipadic}" " -O wakati"
    )
    wakati.parse("")
    return wakati.parse(text).strip().split(" ")


def extract_keyphrase_candidates(text):
    tagger = MeCab.Tagger(f"-r /dev/null -d {Config.mecab.system_dic.ipadic}")
    tagger.parse("")

    node = tagger.parseToNode(text)

    keyphrase_candidates = []
    phrase = []
    phrase_noun = []
    is_adj_candidate = False
    is_multinoun_candidate = False

    while node:
        # adjectives + nouns
        if node.feature.startswith("形容詞"):
            is_adj_candidate = True
            phrase.append(node.surface)
        if node.feature.startswith("名詞") and is_adj_candidate:
            phrase.append(node.surface)
        elif len(phrase) >= 2:
            keyphrase_candidates.append(phrase)

            is_adj_candidate = False
            phrase = []

        # multiple nouns
        if node.feature.startswith("名詞"):
            phrase_noun.append(node.surface)
            is_multinoun_candidate = True
        elif len(phrase_noun) >= 2:
            keyphrase_candidates.append(phrase_noun)

            is_multinoun_candidate = False
            phrase_noun = []
        else:
            is_multinoun_candidate = False
            phrase_noun = []

        node = node.next

    return keyphrase_candidates  # ["".join(cand) for cand in keyphrase_candidates]

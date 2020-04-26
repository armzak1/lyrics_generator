import pandas as pd
import cmudict
import re
import os
import string
from matplotlib import pyplot as plt
from Levenshtein import distance
import itertools

class RhymeFinder:
    def __init__(self, df_lyrics):
        self.df_lyrics = df_lyrics
        self.syllables = cmudict.dict()
    
    def count_syllables(self, word):
        if len(word) < 1:
            return 0
        if word in self.syllables and len(self.syllables[word]) > 0:
            count = [len(list(y for y in x if y[-1].isdigit())) for x in self.syllables[word.lower()]][0]
            if count == 0:
                count += 1
            return count
        else:
            count = 0
            vowels = 'aeiouy'
            word = word.lower()
            if word[0] in vowels:
                count +=1
            for index in range(1,len(word)):
                if word[index] in vowels and word[index-1] not in vowels:
                    count +=1
            if word.endswith('e'):
                count -= 1
            if word.endswith('le') or word.endswith('ye'):
                count += 1
            if count == 0:
                count += 1
            return count
        
    def count_syllables_expression(self, exp):
        return sum([self.count_syllables(word) for word in exp.split(' ')])
    
    def find_all_syllablesets(self, exp):
        words = exp.split(' ')
        rhymesets = []
        for w in words:
            w_syllables = self.syllables[w]
            for version in range(len(w_syllables)):
                w_syllables[version] = [syl for syl in w_syllables[version] if syl[-1].isdigit()]
            rhymesets.append(w_syllables)
        versions = list(itertools.product(*rhymesets))
        res = []
        for v in versions:
            flat_v = ''.join([syl for w in v for syl in w])
            res.append(flat_v)
        return res
    
    def rhyme_similarity(self, exp1, exp2):
        sylset1 = self.find_all_syllablesets(exp1)
        sylset2 = self.find_all_syllablesets(exp2)
        min_dist = 1
        for s1 in sylset1:
            for s2 in sylset2:
                l = max(len(s1), len(s2))
                d = round(distance(s1, s2) / l, 2)
                if d < min_dist:
                    min_dist = d
        return 1 - min_dist
    
    def search_in_line(self, word, lyrics):
        w_nsyl = self.count_syllables_expression(word)
        lines = lyrics.split('\n')
        idx = [i for i, line in enumerate(lines) if line.lower().endswith(word)]
        candidate_idx = []
        rhymes = {}
        for i in idx:
            if i > 0:
                candidate_idx.append(i-1)
            if i < len(lines) - 1:
                candidate_idx.append(i+1)
        for i in candidate_idx:
            n_words = 1
            nsyl = 0
            while nsyl < w_nsyl and n_words < 4:
                exp = ' '.join(lines[i].split(' ')[-n_words:])
                nsyl = self.count_syllables_expression(exp)
                n_words += 1
            rhymes[exp] = self.rhyme_similarity(word, exp)

        return rhymes
    
    def find_lines_ending_with_word(self, word):
        matching_lines = list(self.df_lyrics.apply(lambda x: self.search_in_line(word, x)))
        res = {}
        for d in matching_lines:
            res.update(d)
        res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1])}
        return res
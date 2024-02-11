import pandas as pd
import numpy as np
import csv
from pathlib import Path
from gensim.models import KeyedVectors
import time

from constants import *


class FileExistsException(Exception):
    pass


class InterviewHelper:
    def __init__(
            self,
            run_testing=False,
    ):

        self.run_testing = run_testing

        self.resources_folder = 'resources\\'
        self.vectors_file = self.resources_folder + 'vectors.csv'
        self.vectors_file_short = self.resources_folder + 'vectors_160.csv'
        self.phrases_file = self.resources_folder + 'phrases.txt'

        self.results_folder = 'results\\'

        self.current_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
        self.log_file = self.results_folder + f"Log_File_{self.current_time}.txt"
        Path(self.results_folder).mkdir(parents=True, exist_ok=True)
        file = open(self.log_file, "w")
        file.close()

        self.df_phrases = pd.DataFrame()


    def log_progress(self, msg):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} - {msg}")
        with open(self.log_file, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} - {msg}\n")


    def init_files(self):

        self.log_progress(f"Initializing Interview Helper")

        if self.run_testing:
            vectors_file = self.vectors_file_short
        else:
            vectors_file = self.vectors_file


        if not Path(vectors_file).exists():
            self.log_progress(f"File with Word2Vec vectors was not found at {vectors_file}")

            path = self.resources_folder + 'GoogleNews-vectors-negative300.bin'
            file_name = 'GoogleNews-vectors-negative300.bin'
            location = path + '\\' + file_name

            if not Path(location).exists():
                raise FileExistsException(f"File with Word2Vec vectors was not found at {location}")

            limit = 160 if self.run_testing else 1000000

            self.log_progress(f"Loading Word2Vec model from {location}")
            wv = KeyedVectors.load_word2vec_format(location, binary=True, limit=limit)
            self.log_progress(f"Word2Vec model was successfully loaded from {location}")

            self.log_progress(f"Saving Word2Vec model to {vectors_file}")
            wv.save_word2vec_format(vectors_file)
            self.log_progress(f"Word2Vec model was successfully saved to {vectors_file}")



        dict_words = {}
        self.log_progress(f"Loading Word2Vec model from {vectors_file}")

        if not Path(vectors_file).exists():
            raise FileExistsException(f"File with Phrases was not found at {vectors_file}")

        with open(vectors_file) as f:
            reader = csv.reader(f, delimiter=' ')
            next(reader, None)
            for row in reader:
                if row[0] not in dict_words.keys():
                    dict_words[row[0]] = np.array(row[1:], dtype=np.float32)

        self.dict_words = dict_words

        self.log_progress("Vectors were successfully loaded")

        self.log_progress(f"Loading phrases from {self.phrases_file}")
        dict_phrases = {}
        with open(self.phrases_file, 'r') as f:
            next(f)
            for i, row in enumerate(f.readlines()):
                text_original = row.strip()
                text_clean = text_original.replace('?', '').replace('"', '').lower()
                dict_phrases[i] = {}
                dict_phrases[i][PHRASE_TEXT_ORIG] = text_original
                dict_phrases[i][PHRASE_TEXT_CLEAN] = text_clean
                split_phrase = text_clean.split(' ')
                df_temp = pd.DataFrame()
                for word in split_phrase:
                    word = word.lower()
                    if word in dict_words.keys():
                        df = pd.DataFrame(dict_words[word], columns=[word])
                        df_temp = pd.concat([df_temp, df.transpose()], axis=0)
                if not df_temp.empty:
                    df_temp.loc["Normalized", :] = df_temp.apply(np.linalg.norm, axis=0)
                    dict_phrases[i][PHRASE_VECTOR] = df_temp.loc["Normalized"].to_numpy()
                else:
                    dict_phrases[i][PHRASE_VECTOR] = np.zeros(300, dtype=np.float32)

        self.log_progress("Phrases were successfully loaded")
        self.dict_phrases = dict_phrases


    def _create_phrases_dataframe(self, df_phrases, dict_phrases=None):
        if dict_phrases is None:
            dict_phrases = self.dict_phrases

        for i in df_phrases.index:
            for j in df_phrases.columns:
                df_phrases.loc[i, j] = np.linalg.norm(dict_phrases[j][PHRASE_VECTOR] - dict_phrases[i][PHRASE_VECTOR])

        return df_phrases


    def print_comparison_of_phrases(self):

        self.log_progress(f"Calculating comparison of phrases")

        df_phrases = pd.DataFrame(index=self.dict_phrases.keys(), columns=self.dict_phrases.keys())
        df_phrases = self._create_phrases_dataframe(df_phrases)

        file_name = self.results_folder + f'df_phrases_{self.current_time}.xlsx'

        self.log_progress(f"Writing results to {file_name}")

        cols = [self.dict_phrases[i][PHRASE_TEXT_ORIG] for i in df_phrases.columns]
        df_phrases.columns = cols

        index_names = [self.dict_phrases[i][PHRASE_TEXT_ORIG] for i in df_phrases.index]

        df_phrases.insert(0, "Phrases", index_names)
        df_phrases.to_excel(file_name, index=False)


    def print_comparison_of_user_custom_phrase(self, phrase):

        self.log_progress(f"Calculating comparison of custom phrase: {phrase}")

        new_val_of_phrase = len(self.dict_phrases.keys())
        df_phrases = pd.DataFrame(index=self.dict_phrases.keys(), columns=[new_val_of_phrase])
        dict_phrases = self.dict_phrases.copy()

        text_original = phrase
        text_clean = text_original.replace('?', '').replace('"', '').lower()
        dict_phrases[new_val_of_phrase] = {}
        dict_phrases[new_val_of_phrase][PHRASE_TEXT_ORIG] = text_original
        dict_phrases[new_val_of_phrase][PHRASE_TEXT_CLEAN] = text_clean

        split_phrase = text_clean.split(' ')
        df_temp = pd.DataFrame()
        for word in split_phrase:
            word = word.lower()
            if word in self.dict_words.keys():
                df = pd.DataFrame(self.dict_words[word], columns=[word])
                df_temp = pd.concat([df_temp, df.transpose()], axis=0)
            if not df_temp.empty:
                df_temp.loc["Normalized", :] = df_temp.apply(np.linalg.norm, axis=0)
                dict_phrases[new_val_of_phrase][PHRASE_VECTOR] = df_temp.loc["Normalized"].to_numpy()
            else:
                dict_phrases[new_val_of_phrase][PHRASE_VECTOR] = np.zeros(300, dtype=np.float32)

        df_phrases = self._create_phrases_dataframe(df_phrases, dict_phrases=dict_phrases)

        cols = [dict_phrases[i][PHRASE_TEXT_ORIG] for i in df_phrases.columns]
        df_phrases.columns = cols

        index_names = [dict_phrases[i][PHRASE_TEXT_ORIG] for i in df_phrases.index]
        df_phrases.insert(0, "Phrases", index_names)

        file_name = self.results_folder + f'df_custom_phrase_{self.current_time}.xlsx'
        self.log_progress(f"Writing results to {file_name}")
        df_phrases.to_excel(file_name, index=False)

        df_closest_phrase = df_phrases.loc[df_phrases[cols[0]] == df_phrases[cols[0]].min()]

        phrases = df_closest_phrase["Phrases"].tolist()
        if len(phrases) == 1:
            self.log_progress(f"The closest phrase is {phrases} with the distance of "
                              f"{df_closest_phrase.iloc[0, 1]}")
        else:
            self.log_progress(f"The closest phrases are: {phrases} with the distance of "
                              f"{df_closest_phrase.iloc[0, 1]}")





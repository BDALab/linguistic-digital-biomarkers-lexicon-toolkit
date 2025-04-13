import os

import numpy as np
import pandas as pd

from nltk.util import ngrams
from collections import Counter
import docx
from tqdm import tqdm

import stanza

# STANZA tool for language analysis
# https://stanfordnlp.github.io/stanza/

# POS description
# https://universaldependencies.org/u/pos/

# TreeBank
# https://universaldependencies.org/treebanks/cs_pdt/index.html

# In[] Variables

lang = 'cs'

folder_trans = "transcript_pilot_study_auto"
out_file_name = "features_lang_auto.xlsx"

pth_clin_table = 'data/clinical_data.xlsx'

df_clin = pd.read_excel(pth_clin_table, index_col=0).set_index("#id")

POS = "VERB"  # part-of-speech rate you want to focus on and export these words

download_model = True  # set False if you have already downloaded the model (set True when using for the first time)


# In[] Table

def get_table(content, model):  # , language='cs'

    doc = model(content)

    columns = ["sentence", "text", "upos", "class", "xpos", "feats"]
    table = pd.DataFrame(columns=columns)

    n_indexes = 0
    n_unnoun = 0

    for s, sentence in enumerate(doc.sentences):
        for w, word in enumerate(sentence.words):
            n_indexes += 1

            if word.upos in ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"]:
                class_word = "open"
            elif word.upos in ["ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"]:
                class_word = "closed"
            elif word.upos in ["PUNCT", "SYM", "X"]:
                class_word = "other"
            else:
                class_word = "ERROR"
                n_unnoun += 1
                print("Unknown class word!")

            new_row = {"sentence": s + 1,
                       "text": word.text,
                       "upos": word.upos,
                       "class": class_word,
                       "xpos": word.xpos,
                       "feats": word.feats}
            table = table._append(new_row, ignore_index=True)

    return table


# In[] Features

def get_cd(table):
    ocw = (table["class"] == "open").sum()  # open-class word
    ccw = (table["class"] == "closed").sum()  # closed-class word
    if ccw > 1:
        cd = ocw / ccw
    else:
        cd = np.NaN
    return cd


def get_ng(table, n=4):
    list_words = table["class"] != "other"
    n_words = len(table[list_words]["text"])
    ng = 0
    for n in range(2, n + 1):
        word_ngrams = list(ngrams(table[list_words]["text"], n))
        ngram_counts = Counter(word_ngrams)
        for ngram, count in ngram_counts.items():
            if count > 1:
                # print(f"N-gram: {ngram}, Repetition: {count}")
                ng += count - 1
    return ng / n_words


def get_mattr(table, window_size=58):  # 70 or 58 words
    list_words = (table[table["class"] != "other"]["text"]).tolist()
    if len(list_words) > window_size:
        ratio_unique_words = np.zeros(len(list_words) - window_size + 1)
        vocabulary = set()
        for i in range(len(list_words) - window_size + 1):
            window_words = list_words[i:i + window_size]
            n_unique_words = len(set(window_words))
            # print(set(window_words))
            ratio_unique_words[i] = n_unique_words / window_size
            vocabulary.update(window_words)
        out = np.mean(ratio_unique_words)
    else:
        out = np.NaN
    return out


def get_cc(table):
    n_cconj = table[table["upos"] == "CCONJ"].shape[0]
    n_sconj = table[table["upos"] == "SCONJ"].shape[0]
    if (n_cconj + n_sconj) > 0:
        out = n_cconj / (n_cconj + n_sconj)
    else:
        out = np.NaN
    return out


def get_nw(table):
    n_words = []
    for i in range(table["sentence"].iloc[-1]):
        test = table[table["sentence"] == i + 1]
        n_words.append(len(test[test["class"] != "other"]["text"]))
    return np.mean(n_words)


def get_ns(table):
    n_words = len(table[table["class"] != "other"]["text"])
    n_sentences = table["sentence"].iloc[-1]
    return n_sentences / n_words


def get_np(table, table_pass=None, table_clin=None):
    adj_list = table[table["upos"] == "ADJ"]["feats"]
    text_list = table[table["upos"] == "ADJ"]["text"]
    n_sentences = table["sentence"].iloc[-1]
    if n_sentences > 0:
        n_pass = 0
        for adj, text in zip(adj_list, text_list):
            if "Voice=Pass" in adj:
                n_pass += 1
                if (table_pass is not None) and (table_clin is not None):
                    df_temp = pd.DataFrame(table_clin.loc[ID, "group"], index=[ID], columns=["group"])
                    df_temp["text"] = text
                    table_pass = pd.concat([table_pass, df_temp])
        out = n_pass / n_sentences
    else:
        out = np.NaN
    return out, table_pass


def get_rr(table):
    n_nouns = table[table["upos"] == "NOUN"].shape[0]
    n_verbs = table[table["upos"] == "VERB"].shape[0]
    if n_verbs > 0:
        out = n_nouns / n_verbs
    else:
        out = np.NaN
    return out


def get_sc(table):
    n_sconj = table[table["upos"] == "SCONJ"].shape[0]
    n_nouns = table[table["upos"] == "NOUN"].shape[0]
    n_verbs = table[table["upos"] == "VERB"].shape[0]
    n_words = len(table[table["class"] != "other"]["text"])
    wh_list = table[table["upos"] == "PRON"]["feats"]
    n_wh = 0
    for pron in wh_list:
        if "Rel" in pron:
            n_wh += 1
    return (2 * n_sconj + 2 * n_wh + n_nouns + n_verbs) / n_words


def get_psr(table):
    n_words = len(table[table["class"] != "other"]["text"])
    psr = {
        "ADJ": (table[table["upos"] == "ADJ"].shape[0]) / n_words,
        "ADP": (table[table["upos"] == "ADP"].shape[0]) / n_words,
        "ADV": (table[table["upos"] == "ADV"].shape[0]) / n_words,
        "AUX": (table[table["upos"] == "AUX"].shape[0]) / n_words,
        "INTJ": (table[table["upos"] == "INTJ"].shape[0]) / n_words,
        "CCONJ": (table[table["upos"] == "CCONJ"].shape[0]) / n_words,
        "NOUN": (table[table["upos"] == "NOUN"].shape[0]) / n_words,
        "DET": (table[table["upos"] == "DET"].shape[0]) / n_words,
        "PROPN": (table[table["upos"] == "PROPN"].shape[0]) / n_words,
        "NUM": (table[table["upos"] == "NUM"].shape[0]) / n_words,
        "VERB": (table[table["upos"] == "VERB"].shape[0]) / n_words,
        "PART": (table[table["upos"] == "PART"].shape[0]) / n_words,
        "PRON": (table[table["upos"] == "PRON"].shape[0]) / n_words,
        "SCONJ": (table[table["upos"] == "SCONJ"].shape[0]) / n_words,

    }
    return psr


# In[] Prepare output feature table

if download_model:
    nlp = stanza.Pipeline(lang='cs', processors='tokenize,mwt,pos')
else:
    nlp = stanza.Pipeline(lang='cs', processors='tokenize,mwt,pos', download_method=None)

df_out = pd.DataFrame()

# In[] Loop through subjects

file_list = os.listdir(folder_trans)

list_POS = []
df_POS = pd.DataFrame()
df_passive = pd.DataFrame()

for file_name in tqdm(file_list, desc="Extracting features", unit="subject", ncols=100):

    doc = docx.Document(folder_trans + "/" + file_name)

    string_to_remove = "_CZ-AZV-TSK1_1.docx"
    ID = file_name.replace(string_to_remove, "")

    if ID in df_clin.index:

        # In[] Variables

        # Initialize an empty string to store the text
        text = ""

        # Iterate through the paragraphs in the document and concatenate the text
        for paragraph in doc.paragraphs:
            text += paragraph.text

        # In[] Get Table

        df_person = get_table(text, nlp)

        if POS in df_person["upos"].to_list():

            list_POS_subject = df_person[df_person['upos'] == POS]['text'].tolist()
            for n_name_POS in list_POS_subject:
                df_int_temp = pd.DataFrame(df_clin.loc[ID, "group"], index=[ID], columns=["group"])
                df_int_temp["text"] = n_name_POS
                df_POS = pd.concat([df_POS, df_int_temp])
            list_POS += list_POS_subject

        # In[] Get features

        df_out.loc[ID, "CD"] = get_cd(df_person)
        df_out.loc[ID, "NG"] = get_ng(df_person)
        df_out.loc[ID, "MATTR"] = get_mattr(df_person)
        df_out.loc[ID, "CC"] = get_cc(df_person)
        df_out.loc[ID, "NW"] = get_nw(df_person)
        df_out.loc[ID, "NS"] = get_ns(df_person)

        np_param, df_passive = get_np(df_person, df_passive, df_clin)
        df_out.loc[ID, "NP"] = np_param

        df_out.loc[ID, "RR"] = get_rr(df_person)
        df_out.loc[ID, "SC"] = get_sc(df_person)

        PSR = get_psr(df_person)

        df_out.loc[ID, "PSR-ADJ"] = PSR["ADJ"]
        df_out.loc[ID, "PSR-ADP"] = PSR["ADP"]
        df_out.loc[ID, "PSR-ADV"] = PSR["ADV"]
        df_out.loc[ID, "PSR-AUX"] = PSR["AUX"]
        df_out.loc[ID, "PSR-INTJ"] = PSR["INTJ"]
        df_out.loc[ID, "PSR-CCONJ"] = PSR["CCONJ"]
        df_out.loc[ID, "PSR-NOUN"] = PSR["NOUN"]
        df_out.loc[ID, "PSR-DET"] = PSR["DET"]
        df_out.loc[ID, "PSR-PROPN"] = PSR["PROPN"]
        df_out.loc[ID, "PSR-NUM"] = PSR["NUM"]
        df_out.loc[ID, "PSR-VERB"] = PSR["VERB"]
        df_out.loc[ID, "PSR-PART"] = PSR["PART"]
        df_out.loc[ID, "PSR-PRON"] = PSR["PRON"]
        df_out.loc[ID, "PSR-SCONJ"] = PSR["SCONJ"]

    else:
        print("Subject " + ID + " not present in clinical table.")

# In[] Export

df_out.to_excel("data/" + out_file_name)  # save resulted features
df_POS.to_excel("results/" + POS + "_" + folder_trans + ".xlsx")  # save table with all specific POS
df_passive.to_excel("results/passive_voices_" + folder_trans + ".xlsx")  # save table with all passive forms

# -*- coding: utf-8 -*-
import re

character_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                  's', 't', 'u', 'v', 'w', 'x', 'y', 'z',  '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                  '-', ',', ';', '.', '!', '?', ':', '"', '`', '/', '\\', '|', '_', '@', '#', '$', '%', '^',
                  '&', '*', '~', "'", '+', '=', '<', '>', '(', ')', '[', ']', '{', '}', '\n', " "]


def clean_str(string):
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def strip_char(text, list_char):
    newString = ""
    for i in range(len(text)):
        if text[i] in list_char:
            newString += text[i]
        else:
            newString += "a"  # replace character that not in the char list
    return newString


def mapped_text(x):
    x_new = []
    for i in range(len(x)):
        x_clean = clean_str(x[i])
        x_strip = strip_char(x_clean, character_list)
        x_new.append(x_strip)
    return x_new


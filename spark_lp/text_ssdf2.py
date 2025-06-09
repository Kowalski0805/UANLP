from .choices import Lang
from .utils import split_to_sentences, split_to_words, normalize_sent, \
    filter_stop_words, get_stop_words

def split_sentences(text):
    return split_to_sentences(text)

def split_words(sents):
    return list(map(lambda sent: split_to_words(sent), sents))

def norm_sent(sents):
    return list(map(lambda sent: normalize_sent(sent, Lang("uk")), sents))

def filter_stop(sents):
    lang = Lang('uk')
    return list(map(lambda sent: filter_stop_words(sent, get_stop_words(lang), lang), sents))

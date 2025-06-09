import re

import pandas as pd
from pyspark.sql.functions import col, udf, current_timestamp
from pyspark.sql.pandas.functions import pandas_udf, PandasUDFType

from .choices import Lang
from .utils import flat_map, split_to_sentences, split_to_words, normalize_sent, \
    filter_stop_words, parse_sent, parse_obj_to_dict, tokenize_sent, \
    get_stop_words, cos_sim, morphs
from pyspark.sql.types import *


@udf(returnType=ArrayType(ArrayType(StringType())))
def process_udf(text):
    # lang = Lang(detect(text))
    lang = Lang('uk')
    split_sents = split_to_sentences(text)
    split_words = list(map(split_to_words, split_sents))
    # TODO: test perf without using external libs (like pymorphy3)
    norm_sents = list(map(lambda sents: normalize_sent(sents, lang), split_words))
    # norm_sents = split_words
    # token_sents = list(map(lambda sents: tokenize_sent(sents, lang), norm_sents))
    return list(map(lambda sents: filter_stop_words(sents, get_stop_words(lang), lang), norm_sents))


text_separators = r"[.!?…]+"  # or import if you defined it elsewhere
sent_separators = r"[^\w’ʼґєіїа-яА-Яa-zA-Z0-9]+"  # Ukrainian word split


@pandas_udf(ArrayType(ArrayType(StringType())))
def process_pandas_udf(texts: pd.Series) -> pd.Series:
    lang = Lang.UK
    stop_words = get_stop_words(lang)

    def process(text: str):
        if pd.isna(text) or not text.strip():
            return []

        # Clean text
        text = re.sub('\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Split to sentences
        sentences = [s for s in re.split(text_separators, text) if s]

        # For each sentence
        result = []
        for sent in sentences:
            words = [w for w in re.split(sent_separators, sent) if w]
            normed = [morphs[lang].parse(w.lower())[0].normal_form for w in words]
            filtered = [w for w in normed if w not in stop_words]
            result.append(filtered)

        return result

    return texts.apply(process)

from typing import Union, Set, List

from pyspark.mllib.feature import HashingTF, IDF, IDFModel
from pyspark.ml.feature import CountVectorizer
from pyspark import RDD, SparkContext
from pyspark.mllib.linalg import SparseVector
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, udf, current_timestamp

from spark_lp.choices import Lang
from spark_lp.itext import IText
from spark_lp.text import Text
from spark_lp.utils import flat_map, split_to_sentences, split_to_words, normalize_sent, \
    filter_stop_words, parse_sent, parse_obj_to_dict, tokenize_sent, \
    get_stop_words, cos_sim
from langdetect import detect
from pyspark.sql.types import *
import networkx as nx


class TextDataFrame(IText):
    def __init__(self, spark: SparkSession, text: DataFrame):
        self.text: DataFrame = text
        self._origin_sents: Union[DataFrame, None] = None
        self._sents: Union[DataFrame, None] = None
        self._words_info: Union[DataFrame, None] = None
        self._words: Union[DataFrame, None] = None
        self._tf = None
        self.idf: Union[IDFModel, None] = None
        self._tfidf = None
        self.spark = SparkSession(spark)
        # lang = detect(text)
        # if lang not in ['uk', 'ru']:
        #     lang = 'uk'
        # self.lang: Lang = Lang(lang)
        self.lang = Lang('uk')

    @property
    def sentences(self):
        return self._sents

    @property
    def words_info(self):
        return self._words_info

    @property
    def words(self):
        return self.text

    def split_to_sentences(self):
        split_sents = udf(
            lambda e: split_to_sentences(e, is_cleaned=False),
            ArrayType(StringType())
        )
        sents = self.text.withColumn("body_vec", split_sents("body")).withColumn("title_vec", split_sents("title"))
        self._origin_sents = sents
        split_words = udf(lambda sentences: list(map(lambda sentence: split_to_words(sentence), sentences)))
        self.text = sents.withColumn("body_vec", split_words("body_vec")).withColumn("title_vec", split_words("title_vec"))

    def tokenize(self):
        lang = self.lang
        norm_sent = udf(lambda sents: list(map(lambda sent: normalize_sent(sent, lang), sents)))
        token_sent = udf(
            lambda sents: flat_map(lambda sent: tokenize_sent(sent, lang), sents),
            ArrayType(StructType([
                StructField("number", StringType()),
                StructField("gender", StringType()),
                StructField("pos", StringType()),
                StructField("normal_form", StringType()),
                StructField("word", StringType()),
                StructField("case", StringType())
            ]))
        )
        reduce_sent = udf(lambda sents: list(map(lambda word: word['normal_form'], sents)))

        self.text = self.text.withColumn("body_vec", norm_sent("body_vec")).withColumn("title_vec", norm_sent("title_vec"))
        self.text = self.text.withColumn("body_wvec", token_sent("body_vec")).withColumn("title_wvec", token_sent("title_vec"))
        self.text = self.text.withColumn("body_wvec", reduce_sent("body_wvec")).withColumn("title_wvec", reduce_sent("title_wvec"))

    def filter_stop_words(self, stop_words=None):
        lang = self.lang
        stop_words = stop_words or get_stop_words(self.lang)
        filter_words = udf(lambda sents: list(map(lambda sent: filter_stop_words(sent, stop_words, lang), sents)))
        stop = udf(lambda sents: list(filter(lambda word: word not in stop_words, sents)))

        self.text = self.text.withColumn("body_vec", filter_words("body_vec")).withColumn("title_vec", filter_words("title_vec"))
        self.text = self.text.withColumn("body_wvec", stop("body_wvec")).withColumn("title_wvec", stop("title_wvec"))

    def process(self):
        self.split_to_sentences()
        self.tokenize()
        self.filter_stop_words()
        return self

    def sumarize(self, rdd):
        vertices = rdd.zipWithIndex()
        vertices_df = vertices.toDF(['words', 'id'])
        pairs = vertices.cartesian(vertices).filter(
            lambda pair: pair[0][1] < pair[1][1])
        self.tfidf(rdd)
        tfidfs = self._tfidf
        edges = pairs.map(lambda pair: (
            pair[0][1], pair[1][1], cos_sim(tfidfs[pair[0][1]],
                                            tfidfs[pair[1][1]])
        ))
        g = nx.Graph()
        g.add_weighted_edges_from(edges.collect())
        pr = nx.pagerank(g)
        res = sorted(((i, pr[i], s) for i, s in enumerate(rdd.collect()) if i in pr),
               key=lambda x: pr[x[0]], reverse=True)
        print('\n'.join([str(r) for r in res]))
        # edges_df = edges.toDF(['src', 'dst', 'weight'])
        #
        # graph = GraphFrame(vertices_df, edges_df)
        # ranked_sents = graph.pageRank(resetProbability=0.15, tol=0.01)
        # print(vertices.collect())
        # ranked_sents.vertices.show(truncate=False)
        # print(ranked_sents.vertices.select(['id', 'pagerank']).rdd.sortBy(lambda row: row.pagerank).collect())

    def tfidf(self, rdd):
        tf = HashingTF().transform(rdd)
        self._tf = tf
        tf.cache()
        idf = IDF().fit(tf)
        self.idf = idf
        tfidf = idf.transform(tf)
        self._tfidf = dict(enumerate(tfidf.collect()))

    @staticmethod
    def get_tfidf(idf, sentence: List[str]) -> SparseVector:
        tf = HashingTF().transform(sentence)
        return idf.transform(tf)

class TextsCorpus:
    def __init__(self, sc: SparkContext, texts: Union[List, RDD]):
        self.sc: SparkContext = sc
        self.texts: RDD = self._tokenize_texts(sc, texts)
        self.idf = self._compute_idf(self.texts)

    @staticmethod
    def _tokenize_texts(sc: SparkContext, texts: Union[List[str], RDD]):
        if isinstance(texts, list):
            return sc.parallelize([Text(text).process().words for text in texts])
        else:
            return texts.map(lambda text: Text(text).process().words)

    @staticmethod
    def _compute_idf(texts: RDD) -> IDFModel:
        tf = HashingTF().transform(texts)
        tf.cache()
        idf = IDF().fit(tf)
        return idf

    def get_tfidf(self, text_str) -> SparseVector:
        tf = HashingTF().transform(Text(text_str).process().words)
        return self.idf.transform(tf)

    def extend(self, texts: List[str]):
        self.texts = self.texts.union(self._tokenize_texts(self.sc, texts))
        self.idf = self._compute_idf(self.texts)

    def get_similarity(self, text1: str, text2: str) -> float:
        v1 = self.get_tfidf(text1)
        v2 = self.get_tfidf(text2)
        return cos_sim(v1, v2)

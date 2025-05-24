import time
import cudf
from random import choices

# Adjust delimiter and encoding if needed (e.g., 'utf-8-sig' or 'cp1251')
morph_df = cudf.read_csv("dict_corp_lt.txt", sep=" ", names=["surface", "lemma", "tags"], header=None)
df = cudf.read_json("data_jsonl_single/combined.json", lines=True)

import cupy as cp

def tokenize(text_col):
    # lowercase and simple tokenization
    text_col = text_col.str.lower()
    return text_col.str.replace_tokens(["\n", "\r", "\t"], " ").str.replace(" +", " ", regex=True).str.split()

start = time.perf_counter()
# select just link and body
df = df[["link", "body"]]
df["tokens"] = tokenize(df["body"])
df = df[["link", "tokens"]]
df = df.explode("tokens")  # One word per row
df = df.rename(columns={"tokens": "surface"})
df = df.merge(morph_df, on="surface", how="left")
df["lemma"] = df["lemma"].astype("str")
df["surface"] = df["surface"].astype("str")
df["normalized_token"] = df["lemma"].fillna(df["surface"])
print(cudf.DataFrame(
    {'a': ['1', '1', '1', '2', '2'], 'b': [1, 1, 2, 2, 3], 'c': [1, 2, 3, 4, 5]}
).dtypes)
df = df.groupby(["link"]).apply(lambda x: x["normalized_token"].tolist()).reset_index()

print(df)
end = time.perf_counter()
print(f"⏱ Time taken: {end - start:.2f} seconds")

# def lemmatize_words(words: list[str]) -> cudf.DataFrame:
#     df_input = cudf.DataFrame({"surface": words})
#     df_result = df_input.merge(df_dict, on="surface", how="left")
#     return df_result  # columns: surface, lemma, tags

# # Example usage
# words = ["аакуватого", "аакуватій", "аакуватий", "невідоме_слово"]
# result = lemmatize_words(words)
# end = time.time()
# print(f"⏱ Time taken: {end - start:.2f} seconds")
# print(result)

# start = time.perf_counter()
# gpu_result = lemmatize_words(sample_words)
# gpu_time = time.perf_counter() - start
# print(f"GPU: {len(sample_words)} words in {gpu_time:.4f} seconds ({len(sample_words)/gpu_time:.2f} words/sec)")
#
#
# import pandas as pd
#
# # Load the same file into pandas
# df_dict_cpu = pd.read_csv("dict_corp_lt.txt", sep=" ", names=["surface", "lemma", "tags"], header=None)
#
# def lemmatize_words_cpu(words: list[str]) -> pd.DataFrame:
#     df_input = pd.DataFrame({"surface": words})
#     df_result = df_input.merge(df_dict_cpu, on="surface", how="left")
#     return df_result
#
# start = time.perf_counter()
# cpu_result = lemmatize_words_cpu(sample_words)
# cpu_time = time.perf_counter() - start
# print(f"CPU: {len(sample_words)} words in {cpu_time:.4f} seconds ({len(sample_words)/cpu_time:.2f} words/sec)")


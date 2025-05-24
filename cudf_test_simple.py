import time
import cudf
from random import choices

# Adjust delimiter and encoding if needed (e.g., 'utf-8-sig' or 'cp1251')
df_dict = cudf.read_csv(
    "dict_corp_lt.txt",
    sep=" ",
    names=["surface", "lemma", "tags"],
    header=None,
)

# Load the surface forms from your dictionary
surface_words = df_dict["surface"].to_pandas().tolist()

# Sample 100,000 random words for benchmarking
sample_words = choices(surface_words, k=100_000)
def lemmatize_words(words: list[str]) -> cudf.DataFrame:
    df_input = cudf.DataFrame({"surface": words})
    df_result = df_input.merge(df_dict, on="surface", how="left")
    return df_result  # columns: surface, lemma, tags

# # Example usage
# words = ["аакуватого", "аакуватій", "аакуватий", "невідоме_слово"]
# result = lemmatize_words(words)
# end = time.time()
# print(f"⏱ Time taken: {end - start:.2f} seconds")
# print(result)

start = time.perf_counter()
gpu_result = lemmatize_words(sample_words)
gpu_time = time.perf_counter() - start
print(f"GPU: {len(sample_words)} words in {gpu_time:.4f} seconds ({len(sample_words)/gpu_time:.2f} words/sec)")


import pandas as pd

# Load the same file into pandas
df_dict_cpu = pd.read_csv("dict_corp_lt.txt", sep=" ", names=["surface", "lemma", "tags"], header=None)

def lemmatize_words_cpu(words: list[str]) -> pd.DataFrame:
    df_input = pd.DataFrame({"surface": words})
    df_result = df_input.merge(df_dict_cpu, on="surface", how="left")
    return df_result

start = time.perf_counter()
cpu_result = lemmatize_words_cpu(sample_words)
cpu_time = time.perf_counter() - start
print(f"CPU: {len(sample_words)} words in {cpu_time:.4f} seconds ({len(sample_words)/cpu_time:.2f} words/sec)")


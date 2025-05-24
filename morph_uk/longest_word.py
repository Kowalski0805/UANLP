import pandas as pd

df = pd.read_csv("../ukr_morph_dict.csv")
all_words = pd.concat([df["wordform"], df["lemma"]]).dropna().astype(str)
longest = all_words.map(len).max()
longest_words = all_words[all_words.map(len) == longest].unique()

print(f"Longest word length: {longest}")
print("Examples:")
for w in longest_words:
    print("-", w)

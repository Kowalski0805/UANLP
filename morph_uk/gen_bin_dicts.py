import pandas as pd

df = pd.read_csv("../ukr_morph_dict.csv")
pairs = df[['wordform', 'lemma']].drop_duplicates()

MAX_WORD_LEN = 32

def pad(s):
    return s[:MAX_WORD_LEN].ljust(MAX_WORD_LEN, '\0')

keys = [pad(w) for w in pairs['wordform']]
vals = [pad(l) for l in pairs['lemma']]

with open("dict_keys.bin", "wb") as f:
    for k in keys:
        f.write(k.encode("utf-8"))

with open("dict_vals.bin", "wb") as f:
    for v in vals:
        f.write(v.encode("utf-8"))

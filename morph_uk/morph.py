import pathlib
from collections import defaultdict

from UANLP.morph_uk.affix import parse_aff_files
from UANLP.morph_uk.lemma import parse_base_lst, parse_lemmas
import re

import pandas as pd


# def group_affix_rules_by_group(flat_rules):
#     grouped = defaultdict(list)
#     for entry in flat_rules:
#         group = entry["group"]
#         condition = entry["condition"]
#         for rule in entry["rules"]:
#             grouped[group].append({
#                 "from": rule["from"],
#                 "to": rule["to"],
#                 "tag": rule.get("tag"),
#                 "comment": rule.get("comment"),
#                 "condition": condition
#             })
#     return grouped

def build_lemma_info(lemmas):
    lemma_info = {}
    for row in lemmas:
        lemma_info[row["lemma"]] = {
            "group": row["pos"],
            "features": row["features"],
            "pos": row["pos"]
        }
    return lemma_info

def reverse_lookup(word, lemma_info, affix_rules):
    results = []

    # for group, rules in affix_rules.items():
    for rule in affix_rules:
        to = rule["to"]
        from_ = rule["from"]

        # 1. Does this rule apply to the word?
        if not word.endswith(to):
            continue

        print(f"Rule: {rule}")

        # 2. Reconstruct potential lemma
        base = word[:-len(to)] + (from_ if from_ != "0" else "")

        # 3. Apply condition to the base (reverse logic)
        if rule["condition"]:
            try:
                if not re.search(rule["condition"].rstrip(":") + r"$", base):
                    continue
            except:
                continue

        # 4. Is it a real lemma?
        if base in lemma_info and lemma_info[base]["group"] == rule["group"]:
            results.append({
                "word": word,
                "lemma": base,
                "group": rule["group"],
                "pos": lemma_info[base]["pos"],
                "features": lemma_info[base]["features"],
                "rule": rule
            })

    return results


affix_rules = parse_aff_files(pathlib.Path("data/affix/"))
df = pd.DataFrame(affix_rules)
input_words = [
"теплому",     # adjective: masc, dat
"ящірки",      # noun: gen sg
"синього",     # adjective: masc, gen
"українська",  # adjective: fem nom
"ходив",       # verb: masc past

                             # More noun forms
"двері",       # noun: nom pl
"чоловіка",    # noun: gen sg
"жінці",       # noun: dat sg
"вікном",      # noun: ins sg
"містах",      # noun: loc pl

                            # Verb forms
"читала",      # verb: fem past
"пишемо",      # verb: 1pl pres
"розмовляєш",  # verb: 2sg pres
"поїхав",      # verb: masc past
"буду",        # verb: 1sg fut

                            # Adjective/participle/etc.
"старіший",    # comparative
"найбільший",  # superlative
"відомому",    # adjective: masc, loc
"знайдену",     # participle/adjective

                   # Random test cases
"невідоме",    # neuter adjective
"новини",      # noun: pl nom/acc
"книжками",    # noun: ins pl
"допомагаючи", # gerund
"бігатимеш"    # verb: 2sg fut
]

# Reorder the affix rules by "to" length
affix_rules.sort(key=lambda x: len(x["to"]), reverse=True)

base_lst = parse_lemmas(pathlib.Path("data/dict/"))
# Build the lemma info
lemma_info = build_lemma_info(base_lst)
# Group the affix rules by group
# reverse_affix_index = build_reverse_affix_index(affix_rules)
# Reverse lookup for each word
results = []

for word in input_words:
    results.append(reverse_lookup(word, lemma_info, affix_rules))
    break

# Print the results
print(results)
print(lemma_info['теплий'])

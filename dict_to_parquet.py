import pandas as pd

def parse_lt_dict(filepath, output_name):
    rows = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue  # Skip malformed lines

            wordform, lemma, tagstring = parts

            # Split tags
            all_tags = tagstring.split(":")
            row = {
                "wordform": wordform,
                "lemma": lemma,
                "pos": all_tags[0] if all_tags else None,
                "animacy": None,
                "gender": None,
                "number": None,
                "case": None,
                "aspect": None,
                "voice": None,
                "mood": None,
                "tense": None,
                "person": None,
                "degree": None,
                "misc_tags": []
            }

            # Distribute other tags (best-effort parsing)
            for tag in all_tags[1:]:
                if tag in {"m", "f", "n", "p"}:
                    row["gender"] = tag
                elif tag.startswith("v_"):  # case
                    row["case"] = tag.replace("v_", "")
                elif tag in {"sing", "plur"}:
                    row["number"] = tag
                elif tag in {"inanim", "anim"}:
                    row["animacy"] = tag
                elif tag in {"perf", "impf"}:
                    row["aspect"] = tag
                elif tag in {"pres", "past", "futr"}:
                    row["tense"] = tag
                elif tag in {"1p", "2p", "3p"}:
                    row["person"] = tag
                elif tag in {"act", "pass"}:
                    row["voice"] = tag
                elif tag in {"ind", "imp"}:
                    row["mood"] = tag
                elif tag in {"comp", "supr"}:
                    row["degree"] = tag
                else:
                    row["misc_tags"].append(tag)

            rows.append(row)

    df = pd.DataFrame(rows)
    df["misc_tags"] = df["misc_tags"].apply(lambda x: ",".join(x) if x else None)
    df.to_csv(output_name+'.csv', index=False)
    # df.to_parquet(output_name+'.parquet', index=False)


parse_lt_dict("dict_corp_lt.txt", "ukr_morph_dict")

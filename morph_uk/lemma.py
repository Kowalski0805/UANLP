import pathlib
import re

def parse_base_lst(filepath):
    entries = []

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # TODO: keep agreement tags like rv_oru
            line = line.split('#')[0].strip()  # Remove comments

            # Split lemma and description
            parts = line.split()
            lemma = parts[0]
            # TODO: refactor code below, consider comparative forms (+cs), etc
            raw_tags = " ".join(parts[1:])

            # Extract POS/features from /… part
            match = re.search(r"/([^\s:]+)", raw_tags)
            pos_features = match.group(1).split(".") if match else []

            # Extract extra features like :imperf:perf
            extras = re.findall(r":([\w]+)", raw_tags)

            # Combine all features
            features = pos_features + extras

            entries.append({
                "lemma": lemma,
                "pos": pos_features[0] if pos_features else None,
                "features": features
            })

    return entries

def parse_lemmas(dirpath: pathlib.Path):
    lemmas = []
    for filepath in dirpath.glob("*.lst"):
        lemmas.extend(parse_base_lst(filepath))
    return lemmas


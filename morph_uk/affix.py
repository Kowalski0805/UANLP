import pathlib
import re
from collections import defaultdict

###
# This module is responsible for parsing affix files from dict_uk.
# .aff files describe the rules for how to modify words based on their affixes.
# key features of .aff files:
# group = group name (e.g., "n20" == noun of 2nd declension)
# subgroup = subgroup name (e.g., "n20.a" == noun of 2nd declension, -a ending in genitive). Subgroups may be combined (e.g. "n20.a.p.ke").
# condition = condition for applying the rule (regex pattern)
# rules = list of rules for the group/subgroup:
#   - from = original affix (regex pattern or string)
#   - to = new affix (string or regex pattern)
#   - comment = optional comment (starts with #)
#   - tag = tag (starts with @)
# Typically, conditions and rules are written in pattern:
# ```
# condition:
# from to # comment @ tag
# ```
# But, sometimes conditions are provided separately for each rule, e.g.:
# ```
# from to condition # comment @ tag
# ```
###

def parse_aff_file_flat(filepath):
    affix_rules = []
    current_group = None
    current_subgroup = None
    current_condition = None

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Detect group or subgroup
            if line.startswith("group "):
                parts = line.split()
                if len(parts) != 2:
                    print("Invalid group line:", line)
                    continue
                current_group = parts[1]
                current_subgroup = None
                current_condition = None
                continue

            if line.startswith("subgroup "):
                parts = line.split()
                if len(parts) != 2:
                    print("Invalid subgroup line:", line)
                    continue
                [group, subgroup] = parts[1].split('.')
                if current_group != group:
                    # print(f"Subgroup {parts[1]} does not match current group {current_group}")
                    pass
                current_group = group
                current_subgroup = subgroup
                current_condition = None
                continue

            # Detect condition line
            if line.endswith(':'):
                current_condition = line.rstrip(":")
                continue

            # Otherwise, it's a rule line
            rule = {
                "group": current_group,
                "subgroup": current_subgroup,
                "condition": current_condition,
                "from": None,
                "to": None,
                "tag": None,
                "comment": None,
            }
            # Extract the tag first
            if "@" in line:
                tag = line.split("@")[1].strip()
                rule["tag"] = tag
                line = line.split("@")[0].strip()

            # Extract the comment
            if "#" in line:
                comment = line.split("#")[1].strip()
                rule["comment"] = comment
                line = line.split("#")[0].strip()

            # Now split the line into from and to
            parts = re.split(r'\s+', line)

            if len(parts) > 3 or len(parts) < 2:
                print("Invalid rule line:", line)
                continue

            if len(parts) >= 2:
                rule["from"] = parts[0]
                rule["to"] = parts[1]

            if len(parts) == 3:
                rule["condition"] = parts[2]

            affix_rules.append(rule)

    return affix_rules

def parse_aff_files(dirpath: pathlib.Path):
    affix_rules = []
    for filepath in dirpath.glob("*.aff"):
        affix_rules.extend(parse_aff_file_flat(filepath))
    return affix_rules

# Currently works with plain string and [] regexes
def build_reverse_affix_index(affix_rules):
    results = []
    for rule in affix_rules:
        if re.search('[^а-я\[\]]', rule["condition"]):
            continue

        # expand regexes like in [бдн]ати to бати, дати, нати
        match = re.search(r"\[(.*?)\]", rule["from"])
        if match:
            # Extract the content inside the brackets
            content = match.group(1)
            chars = content.split("")


            # Split by comma and create regex
            expanded = "[" + "".join(content.split(",")) + "]"
            rule["from"] = rule["from"].replace(match.group(0), expanded)

        group = rule["group"]
        condition = rule["condition"].replace(rule["from"])
        suffix = rule["to"]
        index[suffix].append({
            "from": rule["from"],
            "to": suffix,
            "group": group,
            "condition": condition,
            "tag": rule["tag"],
            "comment": rule["comment"],
        })

    return dict(index)  # convert to normal dict

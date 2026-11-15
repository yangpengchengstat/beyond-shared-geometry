"""Enumerate finite action-design choices; no features, fits, or model calls.

Features and state IDs are held fixed. Each table maps state i to
pi^{-1}(a_g(pi(i))). Pure renaming of features together with tables is a
different operation. Output is a planning artifact, not a scientific result.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path


def conjugate(table, pi):
    inverse = tuple(pi.index(i) for i in range(len(pi)))
    return tuple(tuple(inverse[action[pi[i]]] for i in range(len(pi))) for action in table)


def enumerate_group(name, elements, original):
    original = tuple(tuple(row) for row in original)
    tables = {}
    for pi in itertools.permutations(range(len(original[0]))):
        table = conjugate(original, pi)
        tables.setdefault(table, pi)
    families = {}
    for table, pi in tables.items():
        image = tuple(sorted(set(table)))
        families.setdefault(image, []).append((table, pi))
    output = []
    for image, members in sorted(families.items()):
        contains_original = any(table == original for table, _ in members)
        table, pi = next((pair for pair in members if pair[0] == original), min(members))
        output.append({
            "contains_original_action": contains_original,
            "labelled_tables_in_family": len(members),
            "representative_permutation": list(pi),
            "action_table": [list(row) for row in table],
        })
    return {
        "group": name,
        "elements_in_row_order": elements,
        "number_states": len(original[0]),
        "distinct_labelled_tables": len(tables),
        "distinct_image_families": len(families),
        "nonoriginal_image_families": len(families) - 1,
        "original_action_table": [list(row) for row in original],
        "families": output,
    }


def main():
    c7 = [[(i + g) % 7 for i in range(7)] for g in range(7)]
    elements_d4 = [(k, b) for k in range(4) for b in (0, 1)]
    reflection = [1, 0, 3, 2]
    d4 = [[((reflection[i] if b else i) + k) % 4 for i in range(4)] for k, b in elements_d4]
    output = {
        "purpose": "Combinatorial experiment-design enumeration only",
        "model_or_activation_execution": False,
        "conjugation_convention": "a_g^pi(i)=pi_inverse(a_g(pi(i))); features held fixed",
        "representative_rule": "original table retained for original family; otherwise lexicographic table/permutation",
        "groups": [enumerate_group("C7", list(range(7)), c7), enumerate_group("D4", elements_d4, d4)],
    }
    path = Path(__file__).with_name("CONSISTENT_ACTION_DESIGN.json")
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for group in output["groups"]:
        print(group["group"], group["distinct_labelled_tables"], group["distinct_image_families"])


if __name__ == "__main__":
    main()

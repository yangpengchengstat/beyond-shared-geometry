"""Portable finite-action metadata and leakage checks. No model imports."""
from __future__ import annotations
from collections import deque
from itertools import permutations
from dataclasses import dataclass
import json
import numpy as np
AuditError = ValueError

def action_catalog(domain):
    """Literal finite actions, independent of worker group code."""
    if domain == 'C7':
        elements = tuple(range(7)); identity = 0; generators = (1,)
        apply = lambda g, i: (i + g) % 7
    elif domain in ('D4', 'D4_corner'):
        elements = tuple((k, b) for k in range(4) for b in (0, 1))
        identity = (0, 0); generators = ((1, 0), (0, 1))
        apply = lambda g, i: ((1 - i if g[1] else i) + g[0]) % 4
    elif domain in ('S3', 'S3_role'):
        elements = tuple(permutations(range(3))); identity = (0, 1, 2)
        generators = ((1, 0, 2), (0, 2, 1))
        def apply(g, i):
            assignment = elements[i]
            transformed = tuple(assignment[g.index(role)] for role in range(3))
            return elements.index(transformed)
    else:
        raise AuditError('Unsupported action domain')
    states = 7 if domain == 'C7' else 4 if domain in ('D4', 'D4_corner') else 6
    tables = {g: tuple(apply(g, i) for i in range(states)) for g in elements}
    words = {identity: ()}; queue = deque([identity])
    while queue:
        current = queue.popleft()
        for generator in generators:
            table = tuple(tables[generator][tables[current][i]] for i in range(states))
            following = next(g for g in elements if tables[g] == table)
            if following not in words:
                words[following] = words[current] + (generator,); queue.append(following)
    return elements, identity, generators, tables, words


def paired_targets(rows, tables):
    lookup = {}; ids = set(); partition_keys = {}
    for i, row in enumerate(rows):
        required = ('row_id', 'family_id', 'split', 'state_id', 'paraphrase_id', 'orbit_id')
        if any(key not in row for key in required):
            raise AuditError('Incomplete exported row metadata')
        value = row['state_id']
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise AuditError('State identity must be an integer')
        if row['row_id'] in ids:
            raise AuditError('Duplicate row identity')
        ids.add(row['row_id'])
        # S3 cross blocks intentionally reuse context families across entity
        # strata; a family/orbit pair must still stay in a single partition.
        family_orbit = (str(row['family_id']), str(row['orbit_id']))
        if family_orbit in partition_keys and partition_keys[family_orbit] != row['split']:
            raise AuditError('Family/orbit pair crosses partitions')
        partition_keys[family_orbit] = row['split']
        key = (*family_orbit, str(row['paraphrase_id']), int(row['state_id']))
        if key in lookup:
            raise AuditError('Duplicate context/state row')
        lookup[key] = i
    result = {}
    for g, table in tables.items():
        destination = []
        for row in rows:
            state = int(row['state_id'])
            if not 0 <= state < len(table):
                raise AuditError('Invalid source state')
            key = (str(row['family_id']), str(row['orbit_id']), str(row['paraphrase_id']), table[state])
            if key not in lookup:
                raise AuditError('Incomplete state orbit')
            target = lookup[key]
            if rows[target]['split'] != row['split']:
                raise AuditError('Action pair crosses partitions')
            destination.append(target)
        result[g] = np.asarray(destination, dtype=int)
    return result


@dataclass
class Cell:
    z: np.ndarray
    rows: list
    domain: str
    elements: tuple
    identity: str
    generators: tuple
    tables: dict
    words: dict
    targets: dict
    states: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    test_splits: tuple
    registration: str = "semantic"


def make_cell(z, rows, domain, registration="semantic"):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or len(z) != len(rows) or not np.isfinite(z).all():
        raise ValueError("Invalid finite coordinate array")
    elements, identity, generators, tables, words = action_catalog(domain)
    key = lambda g: json.dumps(g, separators=(",", ":"))
    if registration == "consistent_swap01":
        # Fixed before new fits; same permutation for every element. This is a
        # coherent conjugate action, not independently scrambled target labels.
        permutation = np.arange(len(tables[identity]))
        permutation[0], permutation[1] = 1, 0
        inverse = np.argsort(permutation)
        tables = {g: tuple(int(inverse[table[int(permutation[i])]])
                            for i in range(len(permutation))) for g, table in tables.items()}
    elif registration != "semantic":
        raise ValueError("Unknown action registration")
    tables = {key(g): table for g, table in tables.items()}
    targets = paired_targets(rows, tables)
    split = np.asarray([r["split"] for r in rows])
    permitted = {"train", "validation", "test", "test_joint", "test_context", "test_entity", "reserved", "support", "pilot"}
    if not set(split).issubset(permitted):
        raise ValueError("Unknown partition role")
    if not np.any(split == "train") or not np.any(split == "validation"):
        raise ValueError("Both construction and validation partitions required")
    return Cell(z, rows, domain, tuple(map(key, elements)), key(identity),
                tuple(map(key, generators)), tables,
                {key(g): tuple(map(key, word)) for g, word in words.items()},
                targets, np.asarray([int(r["state_id"]) for r in rows]),
                split == "train", split == "validation",
                tuple(s for s in ("test", "test_joint", "test_context", "test_entity") if np.any(split == s)), registration)


def development_subset(z, rows):
    """Physically remove all evaluated test/support/reserved coordinates."""
    keep = [i for i,r in enumerate(rows) if r["split"] in ("train", "validation", "pilot")]
    return z[keep], [rows[i] for i in keep]

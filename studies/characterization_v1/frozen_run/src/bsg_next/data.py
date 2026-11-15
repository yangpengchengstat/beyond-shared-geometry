"""Prospective prompt populations, exact partitions, and sealed data manifests.

Generated families are independent scenario-tree draws conditional on a fixed
public compositional generator. They are NOT 96 independently authored grammar
roots. ``grammar_root_id`` makes that distinction inspectable. A separately
authored grammar holdout can be imported with ``load_sealed_dataset``.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .groups import DAYS, CORNERS, PERMUTATIONS, get_group, pose_tuple

GENERATOR_VERSION = "compositional_scenarios_v1"
GRAMMARS = (
    ("In {setting}, {object} has record {code}. {fact} Ignore its {noise} tag.",
     "Record {code} for {object} in {setting} says: {fact} Its {noise} tag is incidental."),
    ("Clerk: inspect {object} at {setting}. Archivist: {fact} Clerk: retain ticket {code}.",
     "At {setting}, an archivist answers a clerk about {object}: {fact} The ticket is {code}."),
    ("Audit {code} distinguishes {object} from a spare item. For {object}, {fact} The spare has a {noise} seal.",
     "The {noise}-sealed spare is irrelevant to audit {code}. The inspected {object} has this fact: {fact}"),
    ("If a record concerns {object}, use the following entry: {fact} Other records at {setting} are excluded.",
     "Only the {object} entry at {setting} applies. Under that condition, {fact} Use reference {code}."),
    ("A note about {object} reads, '{fact}' The note belongs to {setting}; its reference is {code}.",
     "Reference {code} quotes the {setting} note for {object}: '{fact}' Its border is {noise}."),
    ("Notice {code}, {setting}: {fact} This notice concerns {object}, not the {noise} container.",
     "For {object}, consult notice {code} from {setting}. It states: {fact} Disregard the {noise} container."),
    ("Before archiving {object}, an operator writes {fact} Afterward, the operator adds identifier {code}.",
     "Identifier {code} was added after the operator recorded this about {object}: {fact} The archive is {setting}."),
    ("Two entries discuss {object}. Discard the {noise} draft; the approved entry says {fact} Approval code: {code}.",
     "The approved entry {code} states {fact} It concerns {object}; a {noise} draft was rejected."),
    ("Inspect {object}; read its factual entry, {fact} Then file it at {setting} under {code}.",
     "To file {object} under {code}, first use this entry: {fact} The filing location is {setting}."),
    ("Register | item: {object} | location: {setting} | entry: {fact} | id: {code}.",
     "Id {code} indexes a register at {setting}. Item: {object}. Entry: {fact} Auxiliary tag: {noise}."),
    ("Which entry belongs to {object}? The response is {fact} Verification takes place at {setting}, ticket {code}.",
     "At {setting}, ticket {code} asks about {object}. The verified response states {fact} The tag is {noise}."),
    ("Except for an unrelated {noise} item, the {setting} report concerns {object}. Its entry is {fact} Case {code}.",
     "Case {code} concerns {object}; an unrelated item is excluded. The {setting} report gives {fact}"),
)
SETTINGS = ("the observatory", "the archive", "the workshop", "the harbor", "the gallery", "the laboratory",
            "the warehouse", "the station", "the library", "the greenhouse", "the museum", "the depot")
OBJECTS = ("the instrument", "the parcel", "the display", "the specimen", "the folder", "the marker",
           "the device", "the sample", "the panel", "the ledger", "the exhibit", "the sensor")
NOISE = ("amber", "violet", "silver", "copper", "linen", "ceramic", "velvet", "wooden")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _rng(seed, namespace):
    return random.Random(int(_hash([int(seed), namespace])[:16], 16))


def family_bank(seed=2027101700, test_families=32, pilot=False, native=False):
    """Stable prefixes of a preordered 64-test-family pool; pilot IDs are disjoint."""
    if int(test_families) not in (32, 48, 64):
        raise ValueError("test_families must be a prespecified power-pool prefix:32,48,64")
    partitions = [("pilot", 12)] if pilot else [("train", 16 if native else 48),
                                                ("validation", 8 if native else 16),
                                                ("test", int(test_families))]
    result = []
    for split, count in partitions:
        for index in range(count):
            family_id = f"{'native' if native else 'structural'}_{split}_{index:03d}"
            rng = _rng(seed, family_id)
            grammar = rng.randrange(len(GRAMMARS))
            ast = {"setting": rng.choice(SETTINGS), "object": rng.choice(OBJECTS),
                   "noise": rng.choice(NOISE), "code": f"R{rng.randrange(10**8):08d}"}
            result.append({"family_id": family_id, "split": split,
                           "grammar_root_id": f"grammar_{grammar:02d}", "grammar_index": grammar,
                           "family_ast": ast, "family_seed": int(_hash([seed, family_id])[:16], 16),
                           "generator_version": GENERATOR_VERSION,
                           "sampling_scope": "independent_scenario_draws_conditional_on_fixed_12_root_generator"})
    return result


def _render(family, paraphrase, fact):
    rendered = GRAMMARS[family["grammar_index"]][int(paraphrase)].format(fact=fact, **family["family_ast"])
    # Every root renders its full scenario identifier; otherwise two conditionals
    # that omit code/noise can collapse to an identical prompt across partitions.
    if family["family_ast"]["code"] not in rendered:
        rendered += " Case reference: " + family["family_ast"]["code"] + "."
    return rendered


def _entity_bank(seed, test_orbits=16, pilot=False):
    if test_orbits not in (16, 24):
        raise ValueError("test_orbits must be a registered16 or24 prefix")
    partitions = [("pilot", 8)] if pilot else [("train", 24), ("validation", 8), ("test", test_orbits)]
    used = set()
    result = []
    syllables = ("la", "mi", "vo", "ren", "ta", "sol", "ki", "na", "fel", "por", "zu", "bel")
    for split, count in partitions:
        for index in range(count):
            orbit_id = f"{'pilot' if pilot else 'main'}_triple_{split}_{index:03d}"
            rng = _rng(seed, orbit_id)
            names = []
            while len(names) < 3:
                name = "".join(rng.choice(syllables) for _ in range(5)).capitalize()
                if name not in used:
                    used.add(name)
                    names.append(name)
            result.append({"orbit_id": orbit_id, "entity_split": split, "entities": names})
    return result


def _cross_split(context, entity):
    if context == entity and context in ("train", "validation", "pilot"):
        return context
    if context == entity == "test":
        return "test_joint"
    if context == "test" and entity == "train":
        return "test_context"
    if context == "train" and entity == "test":
        return "test_entity"
    return "reserved"


def build_structural_dataset(domain, seed=2027101700, test_families=32, pilot=False, test_orbits=16, splits=None):
    group = get_group(domain)
    if group.name == "D4_free8":
        raise ValueError("D4_free8 is a native task, not a legacy structural domain")
    families = family_bank(seed, test_families, pilot)
    orbits = _entity_bank(seed, test_orbits, pilot) if group.name == "S3_role" else [
        {"orbit_id": f"lexical_{group.name}", "entity_split": "shared", "entities": []}]
    records = []
    for fi, family in enumerate(families):
        for paraphrase in range(2):
            for orbit in orbits:
                for state_id, state in enumerate(group.state_names):
                    if group.name == "C7":
                        target, fact = state, f"the designated weekday is {state}."
                    elif group.name == "D4_corner":
                        target, fact = state, f"the marked corner is {state}."
                    else:
                        assignment = PERMUTATIONS[state_id]
                        names = orbit["entities"]
                        target = names[assignment[0]]
                        fact = "; ".join(f"role {role} contains {names[assignment[j]]}" for j, role in enumerate(("alpha", "beta", "gamma"))) + "."
                    text = _render(family, paraphrase, fact) + " Summary:"
                    row_id = f"{group.name}/{family['family_id']}/p{paraphrase}/{orbit['orbit_id']}/s{state_id}"
                    row = {**family, **orbit, "row_id": row_id, "record_id": row_id,
                           "domain": group.name, "task": "structural", "paraphrase_id": paraphrase,
                           "paraphrase": paraphrase, "template_id": fi * 2 + paraphrase,
                           "state_id": state_id, "state": state, "text": text, "target": target,
                           "context_split": family["split"], "readout": "final_token" if group.name == "S3_role" else "span_mean"}
                    if group.name == "S3_role":
                        row["assignment"] = list(PERMUTATIONS[state_id])
                        row["split"] = _cross_split(family["split"], orbit["entity_split"])
                    if splits is None or row["split"] in splits:
                        records.append(row)
    validate_records(records, require_complete_orbits=True)
    return records


def native_word_catalog(domain):
    group = get_group(domain)
    if group.name == "C7":
        dev = [(1,), (6,), (1, 1), (6, 6), (1, 1, 1), (1, 6, 1), (1, 1, 1, 1), (6, 6, 6, 6)]
        transfer = [(1,) * 6, (6,) * 6, (1, 6, 1, 1, 6, 1), (1,) * 7,
                    (6,) * 7, (1, 1, 6, 1, 6, 1, 1), (1,) * 8, (6, 1, 6, 6, 1, 6, 1, 6)]
    elif group.name == "D4_free8":
        r, s = (1, 0), (0, 1)
        dev = [(r,), (s,), (r, r), (r, s), (s, r), (r, r, s), (s, r, r), (r, s, s, r)]
        transfer = [(r, s, r, s, r, s), (r, r, s, r, s, r), (s, r, s, s, r, r),
                    (r, s, r, r, s, r, s), (s, r, r, s, r, s, s),
                    (r, r, r, s, r, s, r, r), (s, r, s, r, r, s, r, s), (r, r, s, s, r, r, s, s)]
    else:
        raise ValueError("Native catalog supports C7 or D4_free8")
    return [{"word_id": f"{group.name}_{split}_{i:02d}", "split": split,
             "elements": list(word), "length": len(word), "reduced_element": group.evaluate_word(word)}
            for split, words in (("development", dev), ("transfer", transfer)) for i, word in enumerate(words)]


def _word_text(group, word):
    if not word:
        return "do nothing"
    if group.name == "C7":
        return ", then ".join("advance one weekday" if g == 1 else "go back one weekday" for g in word)
    return ", then ".join("rotate clockwise by one quarter-turn" if tuple(g) == (1, 0)
                            else "reflect across the vertical image axis" for g in word)


def build_native_dataset(domain, seed=2027101700, test_families=32, pilot=False, factorial=False, splits=None):
    group = get_group("D4_free8" if str(domain).lower() in ("d4", "d4_native", "d4_free8_asymmetric_pose_native") else domain)
    if group.name not in ("C7", "D4_free8"):
        raise ValueError("Native tasks are C7 or D4_free8")
    catalog = native_word_catalog(group.name)
    families = family_bank(seed, test_families, pilot, native=True)
    rows = []
    for fi, family in enumerate(families):
        if splits is not None and family["split"] not in splits:
            continue
        if factorial:
            cases = [(i, catalog[j]) for i in range(group.state_count) for j in range(8)]
        elif family["split"] == "train":
            cases = [((fi * 12 + j) % group.state_count, catalog[(fi + j) % 8]) for j in range(12)]
        else:
            cases = [((fi * 12 + j) % group.state_count, catalog[8 + j]) for j in range(8)] + [
                ((fi * 12 + 8 + j) % group.state_count, catalog[(fi * 4 + j) % 8]) for j in range(4)]
        for paraphrase in range(2):
            for ci, (initial, word) in enumerate(cases):
                # Prefix choice must not depend on source-state parity. With the
                # (k,b) D4 state order, s on every odd source collapses all b=1.
                prefix = [] if fi % 2 == 0 else [group.generators[(fi // 2) % len(group.generators)]]
                edit = group.generators[ci % len(group.generators)]
                desired_intermediate = initial
                initial = group.apply(group.inverse(group.evaluate_word(prefix)), desired_intermediate)
                intermediate = group.apply_word(initial, prefix)
                edited = group.apply(edit, intermediate)
                prefix_element = group.evaluate_word(prefix)
                counterpart_initial = group.apply(group.inverse(prefix_element), edited)
                continuation = word["elements"]
                answer = group.state_names[group.apply_word(intermediate, continuation)]
                counterfactual_answer = group.state_names[group.apply_word(edited, continuation)]
                def make_text(start):
                    source = group.state_names[start]
                    if group.name == "C7":
                        fact = f"the initial weekday is {source}."
                        conventions = "A weekday advance moves to the next day of the standard seven-day week, cyclically."
                    else:
                        fact = f"the initial pose, listed as (NW, NE, SE, SW), is {source}."
                        conventions = "Marks move actively with the square; list every answer in (NW, NE, SE, SW) order."
                    return (_render(family, paraphrase, fact) + " " + conventions +
                            " Execute left to right. Prefix program: " + _word_text(group, prefix) +
                            ". Continue. Remaining program: " + _word_text(group, continuation) +
                            ". Give only the final " + ("weekday" if group.name == "C7" else "ordered pose tuple") + ". Answer:")
                text, counterpart = make_text(initial), make_text(counterpart_initial)
                source = group.state_names[initial]
                start = text.index(source)
                carrier = text.index("Continue")
                row_id = f"native/{group.name}/{family['family_id']}/p{paraphrase}/c{ci:03d}"
                rows.append({**family, "row_id": row_id, "record_id": row_id, "domain": group.name,
                             "task": "native_factorial" if factorial else "native", "paraphrase_id": paraphrase,
                             "paraphrase": paraphrase, "scenario_id": ci, "template_id": fi * 2 + paraphrase,
                             "state_id": initial, "state": source, "initial_state": initial,
                             "intermediate_state": intermediate, "prefix_elements": prefix,
                             "allocation_rule": "cycle intermediate states; derive initial by inverse native prefix; report both marginals",
                             "continuation_elements": continuation, "edit_element": edit,
                             "counterfactual_initial_state": counterpart_initial,
                             "text": text, "counterfactual_text": counterpart,
                             "target": source, "source_span": [start, start + len(source)],
                             "carrier_span": [carrier, carrier + len("Continue")], "carrier_target": "Continue",
                             "answer": answer, "counterfactual_answer": counterfactual_answer,
                             "candidates": list(group.state_names), "word_id": word["word_id"],
                             "word_split": word["split"], "word_length": word["length"],
                             "fit_eligible": family["split"] == "train" and word["split"] == "development",
                             "selection_eligible": family["split"] in ("train", "validation") and word["split"] == "development",
                             "order_disagreement": group.apply_word(edited, continuation) != group.apply(edit, group.apply_word(intermediate, continuation)),
                             "readout": "carrier_span"})
    validate_records(rows)
    return rows


def validate_records(records, require_complete_orbits=False):
    if not records:
        raise ValueError("Empty dataset")
    seen_ids, texts, families, orbits = set(), {}, {}, {}
    for row in records:
        for key in ("row_id", "family_id", "paraphrase_id", "state_id", "split", "domain", "text"):
            if key not in row:
                raise ValueError(f"Missing dataset field {key}")
        if row["row_id"] in seen_ids:
            raise ValueError(f"Duplicate row ID {row['row_id']}")
        seen_ids.add(row["row_id"])
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("Blank prompt")
        context_split = row.get("context_split", row["split"])
        previous = families.setdefault(row["family_id"], context_split)
        if previous != context_split:
            raise ValueError("Family crosses context partitions")
        text_key = (row["domain"], row["text"])
        if text_key in texts and texts[text_key] != row["family_id"]:
            raise ValueError("Exact prompt duplicated across different families")
        texts[text_key] = row["family_id"]
        if require_complete_orbits:
            key = (row["domain"], row["family_id"], row["paraphrase_id"], row.get("orbit_id", "shared"))
            orbits.setdefault(key, set()).add(int(row["state_id"]))
    if require_complete_orbits:
        for key, states in orbits.items():
            if states != set(range(get_group(key[0]).state_count)):
                raise ValueError(f"Incomplete action orbit {key}")
    return {"rows": len(records), "families": len(families),
            "grammar_roots": len({r.get("grammar_root_id", "external_unannotated") for r in records}),
            "sampling_scope": sorted({r.get("sampling_scope", "externally_authored_bank") for r in records})}


def dataset_identity(records):
    return _hash(records)


def seal_dataset(records, path, metadata=None):
    """Publish immutable JSONL plus a content manifest; reject replacement."""
    path = Path(path)
    directory = path if path.suffix != ".jsonl" else path.parent
    rows_path = directory / "rows.jsonl" if path.suffix != ".jsonl" else path
    manifest_path = directory / "manifest.json" if path.suffix != ".jsonl" else rows_path.with_suffix(".manifest.json")
    summary = validate_records(records, require_complete_orbits=all(r.get("task") == "structural" for r in records))
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records).encode()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {"schema_version": 1, "status": "sealed", "rows_path": rows_path.name,
                "rows_sha256": digest, "dataset_sha256": dataset_identity(records),
                "summary": summary, "metadata": metadata or {}, "generator_version": GENERATOR_VERSION}
    directory.mkdir(parents=True, exist_ok=True)
    if rows_path.exists() or manifest_path.exists():
        if not rows_path.exists() or not manifest_path.exists():
            raise ValueError("Partial sealed dataset exists; choose a fresh output path")
        old = json.loads(manifest_path.read_text())
        if old != manifest or hashlib.sha256(rows_path.read_bytes()).hexdigest() != digest:
            raise ValueError("Refusing to overwrite a differently sealed dataset")
        return manifest
    tmp = rows_path.with_suffix(rows_path.suffix + ".partial")
    tmp.write_bytes(payload)
    tmp.replace(rows_path)
    mt = manifest_path.with_suffix(".partial")
    mt.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    mt.replace(manifest_path)
    return manifest


def load_sealed_dataset(path, expected_sha256=None):
    path = Path(path)
    rows_path = path / "rows.jsonl" if path.is_dir() else path
    manifest_path = path / "manifest.json" if path.is_dir() else rows_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise ValueError("External data must have a sealed rows.manifest.json; use import_dataset first")
    manifest = json.loads(manifest_path.read_text())
    data = rows_path.read_bytes()
    if manifest.get("status") != "sealed" or hashlib.sha256(data).hexdigest() != manifest["rows_sha256"]:
        raise ValueError("Dataset seal mismatch")
    rows = [json.loads(line) for line in data.splitlines() if line]
    validate_records(rows)
    digest = dataset_identity(rows)
    if digest != manifest["dataset_sha256"] or (expected_sha256 is not None and expected_sha256 != digest):
        raise ValueError("Dataset identity mismatch")
    return rows


def import_dataset(jsonl_path, output, metadata=None):
    """Import an independently authored bank without assigning fake grammar IDs."""
    rows = [json.loads(line) for line in Path(jsonl_path).read_text().splitlines() if line.strip()]
    if any("grammar_root_id" not in row or "sampling_scope" not in row for row in rows):
        raise ValueError("External bank requires grammar_root_id and sampling_scope provenance")
    return seal_dataset(rows, output, metadata={"source_sha256": hashlib.sha256(Path(jsonl_path).read_bytes()).hexdigest(), **(metadata or {})})


def action_target_indices(records, group=None):
    """Targets never cross context/paraphrase/entity orbits."""
    if group is None:
        group = get_group(records[0]["domain"])
    lookup = {(r["family_id"], r["paraphrase_id"], r.get("orbit_id", "shared"), int(r["state_id"])): i for i, r in enumerate(records)}
    if len(lookup) != len(records):
        raise ValueError("Action index expects one complete structural row per state")
    return {g: [lookup[(r["family_id"], r["paraphrase_id"], r.get("orbit_id", "shared"), group.apply(g, r["state_id"]))] for r in records] for g in group.elements}


def build_study_records(domain, seed=2027101700, max_test_families=64, include_pilot=True, native=False, max_test_orbits=None, **kwargs):
    """Create the maximum candidate pool BEFORE any pilot outcomes are observed."""
    builder = build_native_dataset if native else build_structural_dataset
    if max_test_orbits is not None and not native:
        kwargs["test_orbits"] = max_test_orbits
    records = builder(domain, seed=seed, test_families=max_test_families, pilot=False, **kwargs)
    if include_pilot:
        records += builder(domain, seed=seed, test_families=max_test_families, pilot=True, **kwargs)
    validate_records(records)
    return records


def select_phase_records(records, phase, test_families=32, test_orbits=16):
    """Select sealed rows without regenerating them; test prefix excludes pilot."""
    if phase not in ("development", "confirmation", "test", "selected", "all"):
        raise ValueError("phase must be development/confirmation/test/selected/all")
    if test_families not in (32, 48, 64):
        raise ValueError("Unregistered test-family prefix")
    if test_orbits not in (16, 24):
        raise ValueError("Unregistered held-out-orbit prefix")
    result = []
    for row in records:
        context = row.get("context_split", row["split"])
        entity = row.get("entity_split", "shared")
        selected_entity = entity != "test" or int(row["orbit_id"].rsplit("_", 1)[1]) < test_orbits
        selected_test = context == "test" and int(row["family_id"].rsplit("_", 1)[1]) < test_families
        development = context == "pilot" or (context in ("train", "validation") and entity != "test")
        confirmation = selected_test or (context in ("train", "validation", "support") and entity == "test")
        selected = context in ("train", "validation", "support") or selected_test
        if phase == "all" or (selected_entity and ((phase == "development" and development) or (phase == "confirmation" and confirmation) or (phase == "test" and selected_test) or (phase == "selected" and selected))):
            result.append(row)
    return result


def build_support_dataset(domain="S3", seed=2027101700, test_orbits=16, support_contexts=4):
    """Four new support contexts/state/test-orbit; collect only after freezing F.

    Two new families contain two paraphrases each. Support labels are explicit
    adaptation information. These rows never enter PCA or operator fitting.
    """
    group = get_group(domain)
    if group.name != "S3_role" or support_contexts != 4:
        raise ValueError("Registered support arm is S3 with exactly4 contexts/state")
    families = family_bank(seed + 9173, pilot=True)[:2]
    orbits = [orbit for orbit in _entity_bank(seed, test_orbits, False) if orbit["entity_split"] == "test"]
    rows = []
    for fi, source_family in enumerate(families):
        family = {**source_family, "family_id": f"structural_support_{fi:03d}", "split": "support"}
        for paraphrase in range(2):
            for orbit in orbits:
                for state_id, assignment in enumerate(PERMUTATIONS):
                    names = orbit["entities"]
                    fact = "; ".join(f"role {role} contains {names[assignment[j]]}" for j, role in enumerate(("alpha", "beta", "gamma"))) + "."
                    row_id = f"S3_role/{family['family_id']}/p{paraphrase}/{orbit['orbit_id']}/s{state_id}"
                    rows.append({**family, **orbit, "row_id": row_id, "record_id": row_id,
                                 "domain": "S3_role", "task": "structural_support", "paraphrase_id": paraphrase,
                                 "paraphrase": paraphrase, "template_id": fi * 2 + paraphrase,
                                 "state_id": state_id, "state": group.state_names[state_id], "assignment": list(assignment),
                                 "text": _render(family, paraphrase, fact) + " Summary:", "target": names[assignment[0]],
                                 "context_split": "support", "split": "support", "readout": "final_token",
                                 "adaptation_information": "four_support_contexts_for_every_assignment; no_transformation_refit"})
    validate_records(rows, require_complete_orbits=True)
    return rows


build_support_records = build_support_dataset


def render_native_record(record, initial_state=None, continuation_elements=None, prefix_elements=None, edit_element=None):
    """Render natural state/operation variants without substring replacement."""
    row = json.loads(json.dumps(record))
    group = get_group(row["domain"])
    initial = int(row["initial_state"] if initial_state is None else initial_state)
    prefix = row["prefix_elements"] if prefix_elements is None else list(prefix_elements)
    continuation = row["continuation_elements"] if continuation_elements is None else list(continuation_elements)
    edit = group.normalize(row["edit_element"] if edit_element is None else edit_element)
    intermediate = group.apply_word(initial, prefix)
    edited = group.apply(edit, intermediate)
    counterpart_initial = group.apply(group.inverse(group.evaluate_word(prefix)), edited)
    def text_for(start):
        source = group.state_names[start]
        if "native_prompt_template" in row:
            return row["native_prompt_template"].format(initial_state=source,
                        prefix_program=_word_text(group, prefix), continuation_program=_word_text(group, continuation))
        if group.name == "C7":
            fact, conventions = f"the initial weekday is {source}.", "A weekday advance moves to the next day of the standard seven-day week, cyclically."
        else:
            fact = f"the initial pose, listed as (NW, NE, SE, SW), is {source}."
            conventions = "Marks move actively with the square; list every answer in (NW, NE, SE, SW) order."
        return (_render(row, row["paraphrase_id"], fact) + " " + conventions + " Execute left to right. Prefix program: " +
                _word_text(group, prefix) + ". Continue. Remaining program: " + _word_text(group, continuation) +
                ". Give only the final " + ("weekday" if group.name == "C7" else "ordered pose tuple") + ". Answer:")
    text = text_for(initial)
    source = group.state_names[initial]
    carrier_target = row.get("carrier_target", "Continue")
    if text.count(source) != 1 or text.count(carrier_target) != 1:
        raise ValueError("Native render template needs unique source state and carrier marker")
    start, carrier = text.index(source), text.index(carrier_target)
    row.update({"initial_state": initial, "state_id": initial, "state": source, "target": source,
                "prefix_elements": prefix, "continuation_elements": continuation, "edit_element": edit,
                "intermediate_state": intermediate, "counterfactual_initial_state": counterpart_initial,
                "text": text, "counterfactual_text": text_for(counterpart_initial),
                "source_span": [start, start + len(source)], "carrier_span": [carrier, carrier + len(carrier_target)],
                "answer": group.state_names[group.apply_word(intermediate, continuation)],
                "counterfactual_answer": group.state_names[group.apply_word(edited, continuation)],
                "word_length": len(continuation),
                "order_disagreement": group.apply_word(edited, continuation) != group.apply(edit, group.apply_word(intermediate, continuation))})
    if continuation_elements is not None or prefix_elements is not None or initial_state is not None or edit_element is not None:
        row["row_id"] = row["record_id"] = record["row_id"] + "/variant/" + _hash([initial, prefix, continuation, edit])[:16]
    return row


build_native_records = build_native_dataset


def render_rebinding_record(record, state_id=None, query_role=None):
    """Fixed-token whole-prompt role binding; primary vectors are not concatenated."""
    row = json.loads(json.dumps(record))
    group = get_group("S3")
    state_id = int(row["state_id"] if state_id is None else state_id)
    role = int(row.get("query_role", 0) if query_role is None else query_role)
    if role not in range(3):
        raise ValueError("query_role must be0,1,2")
    names, assignment = row["entities"], PERMUTATIONS[state_id]
    moved = group.apply(group.generators[0], state_id)
    def prompt_for(i):
        fact = "; ".join(f"role {label} contains {names[PERMUTATIONS[i][j]]}" for j, label in enumerate(("alpha", "beta", "gamma"))) + "."
        return _render(row, row["paraphrase_id"], fact) + " Which entity occupies role " + ("alpha", "beta", "gamma")[role] + "? Answer:"
    text = prompt_for(state_id)
    carrier = text.rindex("Answer")
    row.update({"task": "rebinding", "state_id": state_id, "state": group.state_names[state_id],
                "initial_state": state_id, "intermediate_state": state_id, "assignment": list(assignment),
                "query_role": role, "text": text, "counterfactual_text": prompt_for(moved),
                "answer": names[assignment[role]], "counterfactual_answer": names[PERMUTATIONS[moved][role]],
                "candidates": list(names), "target": names[assignment[0]], "carrier_target": "Answer",
                "carrier_span": [carrier, carrier + len("Answer")], "readout": "final_token",
                "prefix_elements": [], "continuation_elements": [], "edit_element": group.generators[0],
                "counterfactual_initial_state": moved,
                "fit_eligible": row["split"] == "train", "selection_eligible": row["split"] in ("train", "validation")})
    if state_id != record["state_id"] or role != record.get("query_role", 0):
        row["row_id"] = row["record_id"] = record["row_id"] + f"/rebind_s{state_id}_q{role}"
    return row


def build_rebinding_records(domain="S3", seed=2027101700, test_families=32, pilot=False, splits=None, query_role=0):
    if get_group(domain).name != "S3_role":
        raise ValueError("Rebinding probe uses S3 entity-to-role assignment")
    records = build_structural_dataset("S3", seed=seed, test_families=test_families, pilot=pilot, splits=splits)
    result = [render_rebinding_record(row, query_role=query_role) for row in records]
    for row in result:
        row["row_id"] = row["record_id"] = "rebinding/" + row["row_id"]
    validate_records(result, require_complete_orbits=True)
    return result


def render_native_sham(record, kind="unrelated_attribute"):
    """Declared task controls, distinct from the zero-displacement hook sham."""
    if kind not in ("list", "initial_state", "unrelated_attribute"):
        raise ValueError("Unknown native task sham")
    row = json.loads(json.dumps(record))
    group = get_group(row["domain"])
    def modify(text, initial):
        prefix = text.rsplit("Give only the final", 1)[0]
        if kind == "initial_state":
            return prefix + "Ignore the program result. Repeat only the initial " + ("weekday" if group.name == "C7" else "ordered pose tuple") + ". Answer:"
        if kind == "list":
            return prefix + "Separate list: copper, linen, violet. Ignore all programs and report the first item in this list. Answer:"
        return prefix + "Ignore all program outputs. Report only the case reference code. Answer:"
    row["text"] = modify(row["text"], row["initial_state"])
    row["counterfactual_text"] = modify(row["counterfactual_text"], row["counterfactual_initial_state"])
    if kind == "initial_state":
        row["answer"] = group.state_names[row["initial_state"]]
        row["counterfactual_answer"] = group.state_names[row["counterfactual_initial_state"]]
        row["candidates"] = list(group.state_names)
    elif kind == "list":
        row["answer"] = row["counterfactual_answer"] = "copper"
        row["candidates"] = ["copper", "linen", "violet"]
    else:
        code = row["family_ast"]["code"]
        row["answer"] = row["counterfactual_answer"] = code
        number = int(code[1:])
        row["candidates"] = [code, f"R{(number + 1) % 10**8:08d}", f"R{(number + 2) % 10**8:08d}"]
    row["task_sham"] = kind
    row["row_id"] = row["record_id"] = record["row_id"] + "/sham/" + kind
    return row


def build_native_sham_records(records, kinds=("list", "initial_state", "unrelated_attribute")):
    return [render_native_sham(row, kind) for row in records for kind in kinds]

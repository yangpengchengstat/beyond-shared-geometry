"""Explicit word-field adaptation of the Hu transport comparison.

The included vocabulary is an authored controlled bank, not the paper's
concreteness/density-stratified 1,000-word dataset. External frozen word banks
can replace it. Source test before/after activations are allowed by this task.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .io import identity, atomic_json, read_json, file_sha256, prepare_run
from .config import model_key

CONTEXTS = {
    "neutral": "Thinking of the following concept: {word}.",
    "category": "Thinking of the conceptual category of {word}.",
    "perception": "Thinking of the sensory and perceptual properties of {word}.",
    "situation": "Thinking of a typical situation involving {word}.",
    "affect": "Thinking of the feelings and associations evoked by {word}.",
    "knowledge": "Thinking of general world knowledge about {word}.",
}


def build_word_records(words, seed=2027101700, train_fraction=.8, validation_fraction=.1, test_fraction=None):
    words = list(words)
    if len(words) < 20 or len(words) != len(set(words)):
        raise ValueError("word-field bank requires at least 20 distinct words")
    test_fraction = 1 - train_fraction - validation_fraction if test_fraction is None else test_fraction
    if min(train_fraction, validation_fraction, test_fraction) <= 0 or not np.isclose(train_fraction + validation_fraction + test_fraction, 1):
        raise ValueError("word train/validation/test fractions must be positive and sum to1")
    order = np.random.default_rng(seed).permutation(len(words))
    ntrain, nvalidation = int(len(words)*train_fraction), int(len(words)*validation_fraction)
    if min(ntrain, nvalidation, len(words)-ntrain-nvalidation) < 2:
        raise ValueError("At least2 words required in every partition")
    split = {int(i): "train" if j < ntrain else "validation" if j < ntrain+nvalidation else "test"
             for j, i in enumerate(order)}
    rows = []
    for i, word in enumerate(words):
        if not isinstance(word, str) or not word.strip() or any(c in word for c in "\n\r"):
            raise ValueError("invalid word-field stimulus")
        for context, template in CONTEXTS.items():
            text = template.format(word=word)
            start = text.rindex(word)
            row = {"domain": "word_field", "word": word, "word_id": i,
                   "context": context, "text": text, "target": word,
                   "source_span": [start, start+len(word)], "state_id": i,
                   "family_id": f"word_{i:05d}", "template_id": context,
                   "paraphrase_id": context, "orbit_id": "words", "split": split[i],
                   "sampling_scope": "controlled authored vocabulary; not original stratified sample"}
            row["row_id"] = row["record_id"] = identity(row)
            rows.append(row)
    return rows


def validate_word_rows(rows):
    from .data import validate_records
    validate_records(rows)
    seen = {}
    for row in rows:
        word = row.get("word_id")
        if word is None or row.get("context") not in CONTEXTS:
            raise ValueError("Word-field rows need word_id and a registered context")
        a, b = row.get("source_span", (-1, -1))
        if not 0 <= a < b <= len(row["text"]) or row["text"][a:b] != row.get("word"):
            raise ValueError("Word-field source span does not match its stimulus")
        entry = seen.setdefault(word, {"split": row["split"], "contexts": set(), "word": row["word"]})
        if entry["split"] != row["split"] or entry["word"] != row["word"]:
            raise ValueError("A word identity crosses partitions or lexical labels")
        if row["context"] in entry["contexts"]:
            raise ValueError("Duplicated word/context row")
        entry["contexts"].add(row["context"])
    if any(value["contexts"] != set(CONTEXTS) for value in seen.values()):
        raise ValueError("Each word requires every registered before/after context")
    if {value["split"] for value in seen.values()} != {"train", "validation", "test"}:
        raise ValueError("Word fields require train, validation, and sealed test words")
    return seen


def _field_training_rank(values):
    singular = np.linalg.svd(values-values.mean(0), compute_uv=False)
    return int(np.sum(singular > singular[0]*1e-12)) if len(singular) and singular[0] else 0


def collect_word_fields(config, output, *, runtime_factory=None):
    from .hf import runtime_from_config
    from .extraction import extract_features
    cfg = config.get("crossmodel", {})
    word_path = Path(cfg["word_file"])
    if not word_path.is_absolute() and not word_path.exists():
        word_path = Path(__file__).resolve().parents[2]/word_path
    vocabulary = read_json(word_path)
    words = vocabulary["words"] if isinstance(vocabulary, dict) else vocabulary
    rows = build_word_records(words, int(config.get("seed",42)), float(cfg.get("train_fraction",.8)),
                             float(cfg.get("validation_fraction",.1)), cfg.get("test_fraction"))
    validate_word_rows(rows)
    prepare_run(output, config, "word_field_extraction", provenance={"words":identity(words),"code":file_sha256(__file__)})
    from .data import seal_dataset
    seal_dataset(rows, Path(output)/"dataset", metadata={"unit":"whole_word_all_contexts", "original_paper_stratification":False})
    atomic_json(Path(output)/"vocabulary_identity.json", {"sha256":identity(words),"words":len(words),
                 "contexts":CONTEXTS,"scope":"adaptation; original vocabulary stratification not reproduced"})
    wanted = set(cfg.get("source_models",[])+cfg.get("target_models",[]))
    manifests, slots = [], {}
    if not wanted:
        raise ValueError("Specify source_models and target_models before word-field collection")
    if not wanted <= {model["id"] for model in config["models"]}:
        raise ValueError("Requested word-field model missing from pinned study model list")
    for model in config["models"]:
        if model["id"] not in wanted:
            continue
        runtime = runtime_factory(model,config.get("runtime",{})) if runtime_factory else runtime_from_config(model,config.get("runtime",{}))
        try:
            depth = cfg.get("depths",[.25,.5,.75])
            count = int(runtime.n_layers)
            if not depth or any(not 0 < float(d) <= 1 for d in depth):
                raise ValueError("Word-field depths must be relative indices in(0,1]")
            layers = sorted({max(1,min(count,round(float(d)*count))) for d in depth})
            slots[model["id"]] = {str(float(d)):max(1,min(count,round(float(d)*count))) for d in depth}
            leaf = Path(output)/model_key(model)
            extract_features(runtime, rows, layers, "span_mean", leaf)
            manifests.append(model_key(model))
        finally:
            if hasattr(runtime,"close"): runtime.close()
    if set(slots) != wanted:
        raise ValueError("Requested word-field model missing from pinned model panel")
    atomic_json(Path(output)/"field_caches.json",{"cache_dirs":manifests,"row_identity":identity(rows),"depth_slots":slots,
                 "cache_paths":"relative_to_field_caches_directory", "word_partitions":{name:sum(r["context"]=="neutral" and r["split"]==name for r in rows) for name in ("train","validation","test")}})
    return {"cache_dirs":manifests,"words":len(words),"rows_per_model":len(rows)}


def analyze_word_fields(config, output, cache_root):
    from .extraction import read_cache
    from .crossmodel import run_crossmodel
    from .structural import fit_train_pca
    cfg = dict(config.get("crossmodel",{}))
    index_manifest = read_json(Path(cache_root)/"field_caches.json")
    entries = index_manifest["cache_dirs"]
    caches = {}
    hashes = {}
    for entry in entries:
        leaf = Path(entry) if Path(entry).is_absolute() else Path(cache_root)/entry
        rows, arrays, manifest = read_cache(leaf)
        validate_word_rows(rows)
        name = manifest.get("model")
        if isinstance(name,dict): name = name.get("id",name.get("model_id"))
        if not name or name in caches:
            raise ValueError("Missing/duplicated word-field model identity")
        caches[name] = (rows,arrays)
        hashes[model_key(name)] = file_sha256(leaf/"manifest.json")
    pairs, projections, unavailable = {}, {}, []
    for source in cfg.get("source_models",[]):
        for target in cfg.get("target_models",[]):
            if source == target: continue
            if source not in caches or target not in caches:
                raise ValueError("Missing requested source/target word-field cache")
            sr, sa = caches[source]; tr, ta = caches[target]
            if [r["row_id"] for r in sr] != [r["row_id"] for r in tr]:
                raise ValueError("cross-model word caches have different row identities")
            words = sorted({r["word_id"] for r in sr})
            index = {(r["word_id"],r["context"]):i for i,r in enumerate(sr)}
            splits = [sr[index[(w,"neutral")]]["split"] for w in words]
            train = np.array([s=="train" for s in splits]); val = np.array([s=="validation" for s in splits]); test = np.array([s=="test" for s in splits])
            if [r["split"] for r in sr] != [r["split"] for r in tr]:
                raise ValueError("Source/target word partitions differ")
            depth_slots = index_manifest.get("depth_slots")
            if not depth_slots or source not in depth_slots or target not in depth_slots:
                raise ValueError("Cache must record explicit relative-depth mappings; do not zip deduplicated layer lists")
            for position, depth in enumerate(cfg.get("depths",[.25,.5,.75])):
                sl = int(depth_slots[source][str(float(depth))]); tl = int(depth_slots[target][str(float(depth))])
                for context in CONTEXTS:
                    if context=="neutral":continue
                    neutral = [index[(w,"neutral")] for w in words]
                    changed = [index[(w,context)] for w in words]
                    sx = np.asarray(sa[sl][changed],float)-np.asarray(sa[sl][neutral],float)
                    ty = np.asarray(ta[tl][changed],float)-np.asarray(ta[tl][neutral],float)
                    srank = _field_training_rank(sx[train])
                    trank = _field_training_rank(ty[train])
                    rank = min(int(cfg.get("rank",16)),int(train.sum())-1,sx.shape[1],ty.shape[1],int(srank),int(trank))
                    key = f"{model_key(source)}__{model_key(target)}__depth{position}__{context}"
                    if rank<1:
                        unavailable.append({"condition":key,"status":"zero_training_field_rank","source_rank":int(srank),"target_rank":int(trank)})
                        continue
                    sm,sb,_ = fit_train_pca(sx,train,rank);tm,tb,_ = fit_train_pca(ty,train,rank)
                    px,py = (sx-sm)@sb,(ty-tm)@tb
                    pairs[key] = {"source":px,"target":py,
                                  "source_displacements":px,"target_displacements":py,
                                  "units":np.asarray(words),
                                  "train_mask":train,"validation_mask":val,"test_mask":test}
                    projections[key] = {"relative_depth":float(depth),"source_layer":sl,"target_layer":tl,
                        "rank_requested":int(cfg.get("rank",16)),"rank_retained":rank,
                        "source_training_rank":int(srank),"target_training_rank":int(trank),
                        "source_center":sm.tolist(),"target_center":tm.tolist(),
                        "source_basis_sha256":identity(sb),"target_basis_sha256":identity(tb),
                        "source_training_word_ids":np.asarray(words)[train].tolist(),
                        "validation_word_ids":np.asarray(words)[val].tolist(),"test_word_ids":np.asarray(words)[test].tolist()}
    if not pairs:
        raise ValueError("No nonzero training word-field conditions; cannot report transport success")
    local = dict(config); local["crossmodel"] = {**cfg,"input_identity":identity(hashes),
        "nulls":int(cfg.get("nulls",cfg.get("null_replicates",1000))),
        "projection":"training-only centered displacement PCA; equation12 evaluated on centered projected fields",
        "wordfields_code_sha256":file_sha256(__file__)}
    result = run_crossmodel(local,output,arrays=pairs)
    atomic_json(Path(output)/"field_provenance.json",{"input_cache_sha256":hashes,
        "scope":"controlled vocabulary, training-only projected fields; not an original-paper numerical reproduction",
        "source_test_before_after_access":True,"target_test_used_for_fit":False,
        "field_definition":"delta=h_context(word)-h_neutral(word)",
        "projection_definition":"p_source=(delta_source-mean_source_train)@basis_source_train; target analogous",
        "mean_semantics":"Equation12 predicts the centered projected field. To reconstruct a full target displacement add target training mean after decoding; the source training displacement mean is not an observed target-test quantity.",
        "conditions":projections,"unavailable_conditions":unavailable,
        "split_unit":"whole_word_across_all_six_contexts","original_vocabulary_stratification_reproduced":False})
    return result


def _legal_automorphisms(group):
    """Abstract-group registrations, distinct from state-image semantic references."""
    if group.name == "C7":
        maps = [[group.elements.index((k*g) % 7) for g in group.elements] for k in range(1, 7)]
    elif group.name.startswith("D4"):
        maps = []
        for r in ((1, 0), (3, 0)):
            for k in range(4):
                s = (k, 1)
                image = []
                for exponent, reflected in group.elements:
                    power = group.identity
                    for _ in range(exponent):
                        power = group.compose(r, power)
                    image.append(group.elements.index(group.compose(power, s if reflected else group.identity)))
                maps.append(image)
    elif group.name == "S3_role":
        maps = [[group.elements.index(group.compose(group.compose(h, g), group.inverse(h))) for g in group.elements] for h in group.elements]
    else:
        raise ValueError("No registered action-automorphism constructor for this group")
    from .crossmodel import validate_action_permutation
    multiplication = np.asarray(group.multiplication_table(), int)
    unique = list(dict.fromkeys(tuple(p) for p in maps))
    return [validate_action_permutation(p, multiplication).tolist() for p in unique]


def analyze_structural_transport(config, output, cache_root):
    """Held-out row transport and held-out intertwiner constraints, no inference.

    Individual R_g are independently fitted on construction rows for ALL g.
    The alignment A sees generator constraints only; held-out constraints are
    therefore not an unseen-action-learning claim. Family/orbit targets stay out
    of PCA, action fitting, alignment fitting, normalization, and selection.
    """
    from .groups import get_group
    from .data import action_target_indices
    from .extraction import read_cache
    from .structural import fit_train_pca
    from .estimators import select_action_ridge, LAMBDA_GRID
    from .crossmodel import run_crossmodel
    root = Path(cache_root)
    index = root/"caches.json"
    paths = read_json(index)["cache_dirs"] if index.exists() else [p.parent for p in root.rglob("manifest.json")]
    cfg = dict(config.get("crossmodel", {}))
    source_names, target_names = cfg.get("source_models", []), cfg.get("target_models", [])
    if not source_names or not target_names:
        raise ValueError("Structural transport needs registered source_models and target_models")
    wanted = set(source_names+target_names)
    model_specs = {model["id"]:model for model in config["models"]}
    q = int(cfg.get("structural_rank", config.get("structural", {}).get("q_primary", 16)))
    projected, hashes, descriptors = {}, {}, {}
    failures = []
    for path in paths:
        leaf = Path(path)
        if not leaf.is_absolute() and not leaf.exists():
            leaf = root/leaf
        rows, arrays, manifest = read_cache(leaf)
        model = manifest.get("model")
        model = model.get("id",model.get("model_id")) if isinstance(model,dict) else model
        if model not in wanted:
            continue
        group = get_group(rows[0]["domain"])
        domain = group.name
        if domain not in ("C7", "D4_corner", "S3_role"):
            continue
        descriptor_key = model_key(model)+"__"+domain
        if descriptor_key in hashes:
            raise ValueError("Duplicate structural model/domain caches; pass merged phases only")
        hashes[descriptor_key] = file_sha256(leaf/"manifest.json")
        train = np.asarray([row["split"] == "train" for row in rows])
        val = np.asarray([row["split"] == "validation" for row in rows])
        test = np.asarray([row["split"] == ("test_joint" if domain == "S3_role" else "test") for row in rows])
        if not train.any() or not val.any() or not test.any():
            raise ValueError("Structural cross-model cache requires construction, validation and held-out rows")
        if {row["family_id"] for i,row in enumerate(rows) if train[i]} & {row["family_id"] for i,row in enumerate(rows) if test[i]}:
            raise ValueError("Held-out context families leak into construction")
        if domain == "S3_role" and ({row["orbit_id"] for i,row in enumerate(rows) if train[i]} & {row["orbit_id"] for i,row in enumerate(rows) if test[i]}):
            raise ValueError("Joint-new lexical orbits leak into structural alignment construction")
        indices = action_target_indices(rows, group)
        registered_layers = model_specs[model].get("layers", model_specs[model].get("hidden_state_indices", []))
        if any(int(layer) not in arrays for layer in registered_layers):
            raise ValueError("Structural cache lacks a registered model-depth readout")
        for slot, layer in enumerate(registered_layers):
            hidden = arrays[int(layer)]
            try:
                center, basis, pca = fit_train_pca(hidden, train, q)
            except ValueError as error:
                failures.append({"model":model,"group":domain,"layer":int(layer),"status":"registered_rank_failure","error":str(error)})
                continue
            z = np.empty((len(rows), q), dtype=np.float64)
            for start in range(0, len(rows), 4096):
                z[start:start+4096] = (np.asarray(hidden[start:start+4096],float)-center)@basis
            train_pairs = {g:(z[train],z[np.asarray(target)[train]]) for g,target in indices.items()}
            validation_pairs = {g:(z[val],z[np.asarray(target)[val]]) for g,target in indices.items()}
            # Selection uses generator transitions only. Free composite R_g are
            # subsequently fitted at that frozen penalty on construction rows.
            _generators, selection = select_action_ridge({g:train_pairs[g] for g in group.generators},
                {g:validation_pairs[g] for g in group.generators}, grid=cfg.get("action_lambda_grid",LAMBDA_GRID))
            from .estimators import fit_ridge
            actions = np.stack([fit_ridge(*train_pairs[g],selection["normalized_lambda"]).coef for g in group.elements])
            key = (model, domain, slot)
            projected[key] = {"rows":rows,"z":z,"actions":actions,"group":group,
                              "train":train,"validation":val,"test":test,"layer":int(layer)}
            descriptors[model_key(model)+"__"+domain+"__slot"+str(slot)] = {
                "pca":pca,"pca_center_sha256":identity(center),"pca_basis_sha256":identity(basis),
                "layer":int(layer),"action_selection":selection,
                "action_fit_scope":"all_element_maps_construction_rows; lambda_selected_on_generator_validation_only"}
    pairs, interpretations = {}, {}
    for source in source_names:
        for target in target_names:
            if source == target:
                continue
            shared = sorted({(domain, slot) for (model,domain,slot) in projected if model == source} &
                            {(domain, slot) for (model,domain,slot) in projected if model == target})
            for domain,slot in shared:
                a,b = projected[(source,domain,slot)],projected[(target,domain,slot)]
                source_ids = [r["row_id"] for r in a["rows"]]
                destination_lookup = {r["row_id"]:i for i,r in enumerate(b["rows"])}
                if set(source_ids) != set(destination_lookup):
                    raise ValueError("Cross-model structural datasets have different row identities")
                order = np.asarray([destination_lookup[row_id] for row_id in source_ids],int)
                if any(identity(row) != identity(b["rows"][j]) for row,j in zip(a["rows"],order)):
                    raise ValueError("Shared row IDs have different prompt/split/action metadata")
                group = a["group"]
                generators = [group.elements.index(g) for g in group.generators]
                heldout = [i for i,g in enumerate(group.elements) if g not in (*group.generators,group.identity)]
                if not heldout:
                    raise ValueError("No nonidentity held-out intertwiner constraints")
                pair = f"{model_key(source)}__{model_key(target)}__{domain}__slot{slot}"
                pairs[pair] = {"source":a["z"],"target":b["z"][order],"train_mask":a["train"],
                    "validation_mask":a["validation"],"test_mask":a["test"],
                    "units":np.asarray([row["family_id"] for row in a["rows"]]),
                    "source_actions":a["actions"],"target_actions":b["actions"],
                    "fit_elements":np.asarray(generators,int),"heldout_elements":np.asarray(heldout,int),
                    "multiplication_table":np.asarray(group.multiplication_table(),int),
                    "legal_action_permutations":_legal_automorphisms(group)}
                interpretations[pair] = {"source_layer":a["layer"],"target_layer":b["layer"],"group":domain,
                    "fit_elements":generators,"heldout_elements":heldout,
                    "heldout_scope":"alignment constraints and evaluation families/orbits; all R_g themselves fitted on construction",
                    "registration_null":"abstract_group_automorphisms_refit_alignment; not semantic-image ranks or p-values",
                    "rank_warning":"singular or low-rank A can satisfy constraints without mechanistic equivalence"}
    if not pairs:
        raise ValueError("No aligned, adequate-rank structural source/target pairs available")
    local = dict(config)
    local["crossmodel"] = {**cfg,"input_identity":identity(hashes),"analysis":"structural_generator_constraint_transport",
                            "wordfields_code_sha256":file_sha256(__file__)}
    result = run_crossmodel(local, output, arrays=pairs)
    atomic_json(Path(output)/"structural_transport_provenance.json",{"input_cache_manifest_sha256":hashes,
        "projection_conditions":descriptors,"pairs":interpretations,"unavailable_conditions":failures,
        "new_model_forwards":0,"source_target_pca_fitted_on_construction_only":True,
        "scope":"heldout predictive/action compatibility; not unique or equivalent mechanism identification"})
    return result

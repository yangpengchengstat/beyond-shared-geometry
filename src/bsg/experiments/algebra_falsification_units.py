"""Pure scientific units for the Algebra Falsification Suite."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from bsg.experiments.canonical_units import evaluate_operators, load_artifact
from bsg.experiments.main_config import CANONICAL_REVISIONS
from bsg.groups import c7, d4
from bsg.metrics.core import normalized_state_defect, relative_frobenius
from bsg.operators.independent import build_pairs


_PERMUTATION_CACHE = {}


def stable_cell_seed(master_seed: int, *parts) -> int:
    payload = json.dumps([int(master_seed), *parts], sort_keys=True, default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def fit_ridge_map(x, y, alpha: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    gram = x.T @ x + float(alpha) * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y).astype(np.float64)


def fit_generators(spec, z, train_mask, alpha: float) -> dict:
    generators = (c7.GENERATOR,) if spec.name == "C7" else (d4.R, d4.S)
    result = {}
    for generator in generators:
        x, y = build_pairs(spec.frame, z, train_mask, generator, spec.transform_state)
        result[generator] = fit_ridge_map(x, y, alpha)
    return result


def d4_canonical_words() -> dict:
    """Shortest canonical row-vector words, derived only through d4 utilities."""
    words = {d4.IDENTITY: ()}
    frontier = [()]
    while frontier:
        word = frontier.pop(0)
        for generator in (d4.R, d4.S):
            candidate = word + (generator,)
            element = d4.row_word_product(candidate)
            if element not in words:
                words[element] = candidate
                frontier.append(candidate)
    if set(words) != set(d4.ELEMENTS):
        raise RuntimeError("canonical D4 word search did not cover D4")
    return words


def word_operator(word, generators: dict, dim: int) -> np.ndarray:
    result = np.eye(int(dim), dtype=float)
    for generator in word:
        result = result @ generators[generator]
    return result


def d4_matched_wrong_words() -> dict:
    words = d4_canonical_words()
    controls = {}
    for target, correct in words.items():
        if target in {d4.IDENTITY, d4.R, d4.S}:
            continue
        candidates = itertools.product((d4.R, d4.S), repeat=len(correct))
        wrong = next(
            candidate for candidate in candidates
            if candidate != correct and d4.row_word_product(candidate) != target
        )
        controls[target] = tuple(wrong)
    return controls


def _orthogonal(dim: int, rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(int(dim), int(dim))))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def _template_error(spec, z, template_id: int, target, operator) -> float:
    mask = spec.frame.template_id.to_numpy() == int(template_id)
    x, y = build_pairs(spec.frame, z, mask, target, spec.transform_state)
    return normalized_state_defect(x @ operator, y)


def generator_composition_rows(spec, artifact: dict, split, *, alpha: float, random_seed: int) -> list[dict]:
    """Module A rows. Only minimal generators are fitted by the focal method."""
    z = artifact["z"]
    direct = artifact["operators"]
    learned = fit_generators(spec, z, split.train_mask, alpha)
    dim = z.shape[1]
    rng = np.random.default_rng(int(random_seed))
    random_generators = {
        generator: _orthogonal(dim, rng)
        for generator in ((c7.GENERATOR,) if spec.name == "C7" else (d4.R, d4.S))
    }
    rows = []
    if spec.name == "C7":
        targets = [(k, tuple([c7.GENERATOR] * k)) for k in range(2, 7)]
        wrong_words = {}
    else:
        canonical = d4_canonical_words()
        targets = [(element, canonical[element]) for element in d4.ELEMENTS if element not in {d4.IDENTITY, d4.R, d4.S}]
        wrong_words = d4_matched_wrong_words()
    for split_name, template_ids in (("validation", split.validation_ids), ("test", split.test_ids)):
        for template_id in template_ids:
            for target, word in targets:
                operators = {
                    "direct_ridge": direct[target],
                    "generator_composed": word_operator(word, learned, dim),
                    "identity": np.eye(dim),
                    "matched_random_generator": word_operator(word, random_generators, dim),
                }
                direct_error = _template_error(spec, z, template_id, target, operators["direct_ridge"])
                for method, operator in operators.items():
                    error = _template_error(spec, z, template_id, target, operator)
                    rows.append({
                        "group": spec.name, "data_split": split_name, "template_id": int(template_id),
                        "target_element": repr(target), "word": _word_label(word), "word_length": len(word),
                        "step": int(target) if spec.name == "C7" else None,
                        "evaluation_role": "unseen_composite", "method": method,
                        "E_gen_eq": error, "generator_generalization_gap": error - direct_error,
                    })
                if spec.name == "D4":
                    wrong = wrong_words[target]
                    rows.append({
                        "group": spec.name, "data_split": split_name, "template_id": int(template_id),
                        "target_element": repr(target), "word": _word_label(wrong), "word_length": len(wrong),
                        "step": None, "evaluation_role": "unseen_composite", "method": "matched_wrong_word",
                        "E_gen_eq": _template_error(spec, z, template_id, target, word_operator(wrong, learned, dim)),
                        "generator_generalization_gap": np.nan,
                    })
    if spec.name == "C7":
        cycle_operator = np.linalg.matrix_power(learned[c7.GENERATOR], 7)
        for split_name, template_ids in (("validation", split.validation_ids), ("test", split.test_ids)):
            for template_id in template_ids:
                mask = spec.frame.template_id.to_numpy() == int(template_id)
                rows.append({
                    "group": "C7", "data_split": split_name, "template_id": int(template_id),
                    "target_element": repr(c7.IDENTITY), "word": "r^7", "word_length": 7, "step": 7,
                    "evaluation_role": "cycle_diagnostic", "method": "generator_composed",
                    "E_gen_eq": normalized_state_defect(z[mask] @ cycle_operator, z[mask]),
                    "generator_generalization_gap": np.nan,
                })
    else:
        ident = np.eye(dim)
        diagnostics = {
            "r4_identity": relative_frobenius(np.linalg.matrix_power(learned[d4.R], 4), ident),
            "s2_identity": relative_frobenius(np.linalg.matrix_power(learned[d4.S], 2), ident),
            "srs_r_inverse": relative_frobenius(
                learned[d4.S] @ learned[d4.R] @ learned[d4.S], np.linalg.matrix_power(learned[d4.R], 3)
            ),
        }
        for relation, error in diagnostics.items():
            rows.append({
                "group": "D4", "data_split": "operator", "template_id": None,
                "target_element": relation, "word": relation, "word_length": None, "step": None,
                "evaluation_role": "relation_diagnostic", "method": "generator_composed",
                "E_gen_eq": error, "generator_generalization_gap": np.nan,
            })
    return rows


def _word_label(word) -> str:
    if word and all(item == c7.GENERATOR for item in word):
        return f"r^{len(word)}"
    return "".join("r" if item == d4.R else "s" for item in word) or "e"


def is_automorphism(spec, permutation: tuple[int, ...]) -> bool:
    elements = spec.elements
    index = {element: i for i, element in enumerate(elements)}
    if permutation[index[spec.identity]] != index[spec.identity]:
        return False
    for g1 in elements:
        for g2 in elements:
            lhs = permutation[index[spec.compose(g1, g2)]]
            rhs_element = spec.compose(elements[permutation[index[g1]]], elements[permutation[index[g2]]])
            if lhs != index[rhs_element]:
                return False
    return True


def identity_preserving_permutations(spec, *, automorphisms: bool | None = None) -> list[tuple[int, ...]]:
    cache_key = (spec.name, automorphisms)
    if cache_key in _PERMUTATION_CACHE:
        return _PERMUTATION_CACHE[cache_key]
    elements = spec.elements
    identity_index = elements.index(spec.identity)
    movable = [i for i in range(len(elements)) if i != identity_index]
    result = []
    for order in itertools.permutations(movable):
        mapping = list(range(len(elements)))
        for source, target in zip(movable, order):
            mapping[source] = target
        value = tuple(mapping)
        auto = is_automorphism(spec, value)
        if automorphisms is None or auto == automorphisms:
            result.append(value)
    _PERMUTATION_CACHE[cache_key] = result
    return result


def _defect_tensors(spec, z, mask, operators) -> tuple[np.ndarray, np.ndarray]:
    elements = spec.elements
    n = len(elements)
    x = np.asarray(z, dtype=float)[np.asarray(mask, dtype=bool)]
    op = np.empty((n, n, n), dtype=float)
    data = np.empty_like(op)
    for a in range(n):
        for b in range(n):
            product = operators[elements[b]] @ operators[elements[a]]
            transformed = x @ product
            for c in range(n):
                op[a, b, c] = relative_frobenius(product, operators[elements[c]])
                data[a, b, c] = normalized_state_defect(transformed, x @ operators[elements[c]])
    return op, data


def permutation_null_rows(spec, artifact: dict, split, *, samples: int, seed: int) -> list[dict]:
    """Module B null, evaluated from fixed operators and latent coordinates."""
    candidates = identity_preserving_permutations(spec, automorphisms=False)
    if not candidates:
        raise ValueError("non-automorphic permutation set is empty")
    rng = np.random.default_rng(int(seed))
    selected = [candidates[i] for i in rng.integers(0, len(candidates), size=int(samples))]
    rows = []
    index = {element: i for i, element in enumerate(spec.elements)}
    products = np.array([[index[spec.compose(g1, g2)] for g2 in spec.elements] for g1 in spec.elements])
    a, b = np.indices(products.shape)
    for split_name, mask in (("validation", split.validation_mask), ("test", split.test_mask)):
        op_tensor, data_tensor = _defect_tensors(spec, artifact["z"], mask, artifact["operators"])
        true_op = float(op_tensor[a, b, products].mean())
        true_data = float(data_tensor[a, b, products].mean())
        for draw, permutation in enumerate(selected):
            p = np.asarray(permutation, dtype=int)
            rows.append({
                "group": spec.name, "data_split": split_name, "permutation_index": draw,
                "permutation": json.dumps(list(map(int, permutation))), "sampling": "with_replacement",
                "true_E_hom_op": true_op, "null_E_hom_op": float(op_tensor[p[a], p[b], p[products]].mean()),
                "true_E_hom_data": true_data, "null_E_hom_data": float(data_tensor[p[a], p[b], p[products]].mean()),
            })
    return rows


def summarize_permutation_null(frame) -> list[dict]:
    rows = []
    for (group, data_split), cell in frame.groupby(["group", "data_split"], dropna=False):
        for metric in ("E_hom_op", "E_hom_data"):
            true = float(cell[f"true_{metric}"].iloc[0])
            null = cell[f"null_{metric}"].to_numpy(dtype=float)
            sd = float(np.std(null, ddof=1))
            rows.append({
                "group": group, "data_split": data_split, "metric": metric,
                "true_relation_defect": true, "null_mean": float(np.mean(null)),
                "null_median": float(np.median(null)), "null_q025": float(np.quantile(null, 0.025)),
                "null_q975": float(np.quantile(null, 0.975)),
                "true_percentile": 100.0 * float(np.mean(null <= true)),
                "empirical_p": (1.0 + float(np.sum(null <= true))) / (len(null) + 1.0),
                "standardized_null_separation": (float(np.mean(null)) - true) / sd if sd > 0 else np.nan,
                "n_permutations": int(len(null)),
            })
    return rows


def _nontrivial_permutation(rng, size: int, valid=None) -> np.ndarray:
    identity = np.arange(int(size))
    for _ in range(10000):
        permutation = rng.permutation(size)
        if not np.array_equal(permutation, identity) and (valid is None or valid(permutation)):
            return permutation
    raise RuntimeError("failed to sample a valid nontrivial permutation")


def _changes_registered_action(spec, permutation) -> bool:
    return any(
        int(permutation[spec.transform_state(s, g)]) != spec.transform_state(int(permutation[s]), g)
        for g in spec.elements if g != spec.identity
        for s in sorted(map(int, spec.frame.state_id.unique()))
    )


def scrambled_training_pairs(spec, z, train_ids, element, *, variant: str, seed: int):
    """Return train-only scrambled pairs while preserving each template vector multiset."""
    if variant not in {"train_only_state_relabel", "template_independent_target_scramble"}:
        raise ValueError(f"unknown scramble variant: {variant}")
    state_ids = sorted(map(int, spec.frame.state_id.unique()))
    lookup = {(int(r.template_id), int(r.state_id)): i for i, r in spec.frame.iterrows()}
    x_rows, y_rows, permutations = [], [], {}
    for template_id in train_ids:
        seed_parts = (seed, variant, spec.name, int(template_id))
        if variant == "template_independent_target_scramble":
            seed_parts += (repr(element),)
        rng = np.random.default_rng(stable_cell_seed(*seed_parts))
        if variant == "train_only_state_relabel":
            permutation = _nontrivial_permutation(rng, len(state_ids), lambda p: _changes_registered_action(spec, p))
            permutations[int(template_id)] = permutation.copy()
            for state in state_ids:
                source = int(permutation[state])
                target = int(permutation[spec.transform_state(state, element)])
                x_rows.append(z[lookup[(int(template_id), source)]])
                y_rows.append(z[lookup[(int(template_id), target)]])
        else:
            if element == spec.identity:
                permutation = np.arange(len(state_ids))
            else:
                permutation = _nontrivial_permutation(rng, len(state_ids))
            permutations[int(template_id)] = permutation.copy()
            for state in state_ids:
                target = int(permutation[spec.transform_state(state, element)])
                x_rows.append(z[lookup[(int(template_id), state)]])
                y_rows.append(z[lookup[(int(template_id), target)]])
    return np.asarray(x_rows), np.asarray(y_rows), permutations


def fit_scrambled_operators(spec, artifact: dict, split, *, variant: str, seed: int, alpha: float):
    operators, permutations = {}, {}
    for element in spec.elements:
        x, y, mapping = scrambled_training_pairs(
            spec, artifact["z"], split.train_ids, element, variant=variant, seed=seed
        )
        operators[element] = fit_ridge_map(x, y, alpha)
        permutations[repr(element)] = mapping
    return operators, permutations


def correspondence_scramble_rows(spec, artifact: dict, split, *, scramble_seeds, alpha: float) -> list[dict]:
    rows = []
    real = evaluate_operators(
        spec, artifact, split, "real_ridge", optimization_seed=None, random_baseline_seed=None
    )
    for row in real:
        row["scramble_variant"] = "real"
        row["scramble_seed"] = None
        rows.append(row)
    for variant in ("train_only_state_relabel", "template_independent_target_scramble"):
        for scramble_seed in scramble_seeds:
            operators, _ = fit_scrambled_operators(
                spec, artifact, split, variant=variant, seed=int(scramble_seed), alpha=alpha
            )
            null_artifact = {**artifact, "operators": operators}
            evaluated = evaluate_operators(
                spec, null_artifact, split, "scrambled_ridge", optimization_seed=None, random_baseline_seed=None
            )
            for row in evaluated:
                row["scramble_variant"] = variant
                row["scramble_seed"] = int(scramble_seed)
                rows.append(row)
    return rows


class HistoricalCheckpointSource:
    """Strict read-only view over immutable main_full_003 structural checkpoints."""

    def __init__(self, root: Path, expected: dict):
        self.root = Path(root).resolve()
        self.expected = dict(expected)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.ledger = json.loads((self.root / "run_ledger.json").read_text(encoding="utf-8"))
        for field in ("run_id", "experiment", "tier", "status", "git_commit", "config_hash"):
            if self.manifest.get(field) != self.expected.get(field):
                raise ValueError(f"historical {field} mismatch")
        if self.manifest.get("git_dirty") is not False:
            raise ValueError("historical source was not clean")
        revisions = {model.get("id"): model.get("requested_revision") for model in self.manifest.get("models", [])}
        if revisions != CANONICAL_REVISIONS:
            raise ValueError("historical immutable model revision panel mismatch")
        if int(self.manifest.get("primary_latent_dimension", -1)) != 16:
            raise ValueError("historical primary latent dimension mismatch")
        if self.manifest.get("template_split") != {"train_templates": 28, "validation_templates": 6, "test_templates": 6}:
            raise ValueError("historical template split mismatch")
        self._snapshot = self._file_snapshot()

    def _file_snapshot(self):
        return {str(path.relative_to(self.root)): (path.stat().st_size, path.stat().st_mtime_ns) for path in self.root.rglob("*") if path.is_file()}

    def assert_unchanged(self) -> None:
        if self._file_snapshot() != self._snapshot:
            raise RuntimeError("immutable historical source changed during AFS")

    def load_cell(self, spec, *, model_key: str, model: str, depth_slot: float, layer: int, split_seed: int, latent_dim: int):
        matches = []
        for key, unit in self.ledger.get("units", {}).items():
            fields = unit.get("fields", {})
            if not key.startswith("structural_base:") or unit.get("status") != "COMPLETE":
                continue
            try:
                observed_depth = round(float(fields.get("depth_slot", -1)), 2)
            except (TypeError, ValueError):
                continue
            wanted = (
                fields.get("model_key") == model_key and fields.get("model") == model
                and fields.get("group") == spec.name and observed_depth == round(float(depth_slot), 2)
                and int(fields.get("layer", -1)) == int(layer) and int(fields.get("split_seed", -1)) == int(split_seed)
                and int(fields.get("latent_dim", -1)) == int(latent_dim)
            )
            if wanted:
                matches.append(unit)
        if len(matches) != 1:
            raise ValueError(f"expected one historical structural cell, found {len(matches)}")
        npz_output = next((o for o in matches[0]["outputs"] if str(o["path"]).endswith(".npz")), None)
        if npz_output is None:
            raise ValueError("historical structural cell has no NPZ artifact")
        path = self.root / "checkpoints" / "structural_base" / Path(npz_output["path"]).name
        if not path.is_file() or path.stat().st_size != int(npz_output["bytes"]):
            raise ValueError(f"historical artifact missing or size mismatch: {path}")
        if _sha256(path) != npz_output["sha256"]:
            raise ValueError(f"historical artifact checksum mismatch: {path}")
        artifact = load_artifact(path, spec.elements, seed=int(split_seed))
        if artifact["z"].shape != (len(spec.frame), int(latent_dim)):
            raise ValueError("historical latent row/order contract mismatch")
        expected_fit_rows = 28 * len(spec.frame.state_id.unique())
        if int(artifact["pca"]._fit_rows) != expected_fit_rows:
            raise ValueError("historical PCA train-row contract mismatch")
        self.assert_unchanged()
        return artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_embedding_architecture(model, expected_architecture: str, expected_blocks: int) -> None:
    model_type = str(getattr(model.config, "model_type", ""))
    if model_type != expected_architecture:
        raise ValueError(f"embedding architecture mismatch: {model_type} != {expected_architecture}")
    if int(getattr(model.config, "num_hidden_layers", -1)) != int(expected_blocks):
        raise ValueError("embedding model block-count mismatch")
    if not callable(getattr(model, "get_input_embeddings", None)):
        raise ValueError("model does not expose input embeddings")

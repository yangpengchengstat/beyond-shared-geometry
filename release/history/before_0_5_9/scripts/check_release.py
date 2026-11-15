#!/usr/bin/env python3
"""Validate the portable publication surface without importing model runtimes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT_FILES = {"README.md", "AGENTS.md", "LICENSE", "pyproject.toml", "requirements.txt", "requirements-gpu.txt", "Makefile", ".gitignore", ".gitattributes", "CITATION.cff", "CHANGELOG.md", "GETTING_STARTED.md", "RESEARCH_DECISIONS.md", "inputs/README.txt"}
ROOT_FILES.update({"REVIEW_DISPOSITION.md", "THEORY_CHECK.md", "DATA_METRIC_MAP.md", "REPRODUCE_CORE_RESULTS.md", "REVISION_REPORT.md", "PRESERVATION_REPORT.md", "COMPILE_CHECK.md", "ORTHOGONAL_INTEGRATION_REPORT.md", "APPENDIX_DEPENDENCY_AUDIT.csv", "SCIENTIFIC_CONTENT_LOCK.md", "STYLE_AND_STRUCTURE_NOTES.md"})
ROOT_DIRS = {"analysis", "review_reproduction", "src", "configs", "tests", "scripts", "docs", "runpod", "paper", "results", "release", "provenance", ".github", "studies", "repair_v2"}
EXCLUDED_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", "checkpoints", "activation_cache", "hf_cache", "run_outputs", "dist", "build", "imports", "incoming", "local_results", ".venv-audit", ".local-deps", ".tooling"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".gguf", ".npy", ".npz", ".pkl", ".pickle", ".parquet", ".feather", ".zip", ".gz", ".tar", ".log", ".aux", ".bbl", ".blg", ".out", ".fls", ".fdb_latexmk"}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".tex", ".sty", ".cls", ".bst", ".bib", ".json", ".yaml", ".yml", ".toml", ".txt", ".csv", ".cff"}
DIST_CLASSES = {"article", "report", "book", "standalone"}
DIST_PACKAGES = set("algorithm algorithmic algorithm2e amsmath amssymb amsfonts amsthm array balance booktabs calc caption cleveref color enumitem eso-pic etoolbox fancyhdr float fontenc forloop geometry graphicx graphics hyperref ifthen inputenc keyval lmodern longtable mathtools microtype multirow natbib newtxmath newtxtext nicefrac pdfpages pgfplots setspace subcaption subfig textcomp times tikz url xurl xcolor xspace xstring xparse accsupp fancyvrb framed colortbl soul tabularx wrapfig titlesec tocbibind flushend cuted stfloats babel".split())
DIST_BST = {"plain", "plainnat", "abbrv", "abbrvnat", "unsrt", "unsrtnat", "alpha"}
DIST_PACKAGES.update({"placeins", "anyfontsize", "fancyvrb", "algorithmicx", "algpseudocode", "lineno", "rotating", "verbatim"})
REF_RE = re.compile(r"\\(input|include|includegraphics\*?|bibliography|bibliographystyle|documentclass|usepackage|RequirePackage)(?:\s*\[[^\]]*\])?\s*\{([^{}]+)\}")
SECRET_RE = re.compile(r"(?:\b(?:hf_|ghp_|github_pat_|sk-proj-)[A-Za-z0-9_-]{20,}\b|\bAKIA[A-Z0-9]{16}\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
PRIVATE_PATH_RE = re.compile(r"(?:/Users/[A-Za-z0-9_.-]+/|/home/[A-Za-z0-9_.-]+/|/private/(?:tmp|var)/|[A-Za-z]:\\Users\\)")


class ReleaseError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ReleaseError(f"unsafe repository-relative path: {value}")
    target = root / path
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ReleaseError(f"path escapes repository: {value}")
    return target


def source_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if rel.parts[0] not in ROOT_DIRS and str(rel) not in ROOT_FILES:
            continue
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES or path.name.startswith(".env") or path.suffix in {".key", ".pem"}:
            continue
        if path.is_symlink():
            raise ReleaseError(f"symlink is not a publication asset: {rel}")
        if path.is_file():
            files.append(path)
    return files


def uncomment_tex(text: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def tex_dependencies(root: Path, main: str = "paper/main.tex") -> set[Path]:
    """Resolve local TeX/BibTeX/graphics closure and declared distribution packages."""
    main_path = relative_path(root, main)
    paper_root = main_path.parent
    if not main_path.is_file():
        raise ReleaseError(f"missing manuscript: {main}")
    pending = [main_path]
    found: set[Path] = set()
    citations: set[str] = set()
    bib_keys: set[str] = set()
    suffixes = {"input": ["", ".tex"], "include": ["", ".tex"], "includegraphics": ["", ".pdf", ".png", ".jpg", ".jpeg", ".eps"], "bibliography": ["", ".bib"], "bibliographystyle": ["", ".bst"], "documentclass": ["", ".cls"], "usepackage": ["", ".sty"], "RequirePackage": ["", ".sty"]}
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(paper_root.resolve()):
            raise ReleaseError(f"manuscript dependency escapes paper directory: {path.name}")
        found.add(path)
        if path.suffix not in {".tex", ".sty", ".cls", ".bst", ".bib"}:
            continue
        text = uncomment_tex(path.read_text(encoding="utf-8"))
        if path.suffix == ".bib":
            bib_keys.update(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", text))
            continue
        if path.suffix == ".tex":
            for keys in re.findall(r"\\(?:cite[a-zA-Z]*|nocite)\*?(?:\s*\[[^\]]*\]){0,2}\s*\{([^{}]+)\}", text):
                citations.update(key.strip() for key in keys.split(",") if key.strip() != "*" and "#" not in key)
        for command, argument in REF_RE.findall(text):
            command = command.rstrip("*")
            for value in argument.split(","):
                value = value.strip()
                if not value or "\\" in value or "#" in value:
                    raise ReleaseError(f"dynamic TeX dependency requires explicit resolution in {path.name}: {value}")
                if Path(value).is_absolute() or ".." in Path(value).parts:
                    raise ReleaseError(f"nonportable TeX dependency in {path.name}: {value}")
                candidates = [base / (value + suffix) for base in (paper_root, path.parent) for suffix in suffixes[command]]
                resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
                if resolved is not None:
                    pending.append(resolved)
                elif command == "documentclass" and value in DIST_CLASSES:
                    pass
                elif command in {"usepackage", "RequirePackage"} and value in DIST_PACKAGES:
                    pass
                elif command == "bibliographystyle" and value in DIST_BST:
                    pass
                else:
                    raise ReleaseError(f"missing {command} dependency in {path.relative_to(root)}: {value}")
    if citations - bib_keys:
        raise ReleaseError("unresolved bibliography keys: " + ", ".join(sorted(citations - bib_keys)))
    return found


def scan_publication_files(root: Path, files: list[Path]) -> None:
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in ROOT_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        if SECRET_RE.search(text):
            raise ReleaseError(f"possible credential/private key in {path.relative_to(root)}")
        if PRIVATE_PATH_RE.search(text):
            raise ReleaseError(f"private absolute filesystem path in {path.relative_to(root)}")


def check_artifacts(root: Path, dependencies: set[Path]) -> int:
    manifest_path = root / "release/artifact_manifest.json"
    if not manifest_path.is_file():
        raise ReleaseError("missing release/artifact_manifest.json")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data.get("artifacts", [])
    declared = {}
    for entry in entries:
        value = entry["path"]
        if value in declared:
            raise ReleaseError(f"duplicate artifact manifest path: {value}")
        path = relative_path(root, value)
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            raise ReleaseError(f"artifact checksum mismatch or missing file: {value}")
        if not entry.get("source_run") or not entry.get("source_path") or not entry.get("role"):
            raise ReleaseError(f"artifact lacks source_run/source_path/role: {value}")
        if Path(entry["source_path"]).is_absolute() or ".." in Path(entry["source_path"]).parts:
            raise ReleaseError(f"artifact source_path must be archive-relative: {value}")
        declared[value] = entry
    required = {str(path.relative_to(root)) for path in dependencies if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".eps"} or "generated" in path.parts}
    for folder in (root / "results/tables", root / "results/figures", root / "results/derived"):
        if folder.exists():
            required.update(str(path.relative_to(root)) for path in folder.rglob("*") if path.is_file() and not path.name.startswith("."))
    if required - declared.keys():
        raise ReleaseError("untraced publication artifacts: " + ", ".join(sorted(required - declared.keys())))
    return len(entries)


def check_source_manifest(root: Path) -> int:
    manifest = root / "release/source_manifest.json"
    if not manifest.is_file():
        raise ReleaseError("missing release/source_manifest.json")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    entries = data.get("files", {})
    if not entries:
        raise ReleaseError("empty frozen numerical source manifest")
    for value, digest in entries.items():
        path = relative_path(root, value)
        if not path.is_file() or sha256(path) != digest:
            raise ReleaseError(f"frozen numerical source changed: {value}")
    return len(entries)


def check_template(root: Path) -> int:
    metadata_file = root / "provenance/template_source.json"
    if not metadata_file.is_file():
        raise ReleaseError("missing provenance/template_source.json")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    entries = metadata.get("support_sha256", {})
    if metadata.get("modified") is not False or not entries:
        raise ReleaseError("official template lacks unchanged support-file provenance")
    venue = json.loads((root / "release/release_manifest.json").read_text()).get("target_venue", "ICML")
    if venue == "TMLR":
        if metadata.get("target_venue") != "TMLR" or "tmlr.sty" not in entries:
            raise ReleaseError("TMLR template lacks official support-file provenance")
        if not (metadata.get("source_url") or metadata.get("url")):
            raise ReleaseError("TMLR template source URL is missing")
    elif metadata.get("template_year") != 2026:
        raise ReleaseError("legacy fallback template year mismatch")
    for name, digest in entries.items():
        path = relative_path(root, "paper/" + name)
        if not path.is_file() or sha256(path) != digest:
            raise ReleaseError(f"official template support file changed: {name}")
    return len(entries)


def check_pdf_qa(root: Path, dependencies: set[Path]) -> bool:
    """Validate a saved PDF QA attestation with only the standard library."""
    qa_path = root / "release/pdf_qa.json"
    if not qa_path.is_file():
        return False
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if qa.get("errors") != []:
        raise ReleaseError("saved PDF QA contains errors or lacks an errors list")
    pdf = relative_path(root, qa.get("pdf_path", "paper/main.pdf"))
    if not pdf.is_file() or sha256(pdf) != qa.get("pdf_sha256"):
        raise ReleaseError("saved PDF QA checksum does not match compiled PDF")
    sources = qa.get("source_sha256", {})
    if not sources:
        raise ReleaseError("saved PDF QA lacks source checksums")
    required = {str(path.relative_to(root)) for path in dependencies}
    if required - sources.keys():
        raise ReleaseError("saved PDF QA omits manuscript dependencies: " + ", ".join(sorted(required - sources.keys())))
    for name, digest in sources.items():
        path = relative_path(root, name)
        if not path.is_file() or sha256(path) != digest:
            raise ReleaseError(f"saved PDF is stale after a source edit: {name}")
    return True


def check_review_coverage(root: Path) -> int:
    """Check the public inventory attestation without requiring private files."""
    coverage_path = root / "release/review_coverage.json"
    inventory_path = root / "release/historical_file_inventory.json"
    if not coverage_path.exists() and not inventory_path.exists():
        return 0
    if not coverage_path.is_file() or not inventory_path.is_file():
        raise ReleaseError("review coverage and historical inventory must be supplied together")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if coverage.get("errors") != [] or not isinstance(inventory, list):
        raise ReleaseError("review coverage reports errors or has an invalid inventory")
    if coverage.get("historical_inventory_sha256") != sha256(inventory_path):
        raise ReleaseError("historical inventory checksum does not match review coverage")
    if coverage.get("files") != len(inventory) or coverage.get("bytes") != sum(row["bytes"] for row in inventory):
        raise ReleaseError("historical inventory count/size does not match review coverage")
    paths = [row["path"] for row in inventory]
    if len(set(paths)) != len(paths):
        raise ReleaseError("historical inventory contains duplicate paths")
    for value in paths:
        # These identify historical files; they need not be included in the release.
        if Path(value).is_absolute() or ".." in Path(value).parts:
            raise ReleaseError("historical inventory contains a nonrelative path")
    conversations = coverage.get("conversations", {})
    if conversations.get("complete_historical_project_coverage") is not False:
        raise ReleaseError("bounded conversation access must not be labeled complete history")
    if coverage.get("all_historical_measurements_rederived") is not False or coverage.get("model_execution_performed") is not False:
        raise ReleaseError("release review must not imply all measurements or models were rerun")
    return len(inventory)


def check_orthogonal_results(root: Path, metadata: dict) -> dict | None:
    """Verify previously completed records and source identities; no matrix code."""
    if not metadata.get("orthogonal_comparator_results_available"):
        return None
    folder = relative_path(root, metadata["orthogonal_comparator_results"])
    binding = json.loads((folder / "SOURCE_BINDINGS.json").read_text())
    checked = 0
    retained_private = []
    for name, entry in binding["copied_files"].items():
        path = relative_path(folder, name)
        if not path.is_file() and path.suffix in EXCLUDED_SUFFIXES:
            retained_private.append(name)
            continue
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ReleaseError("Completed orthogonality evidence changed: " + name)
        checked += 1
    coverage = json.loads((folder / "coverage.json").read_text())
    tests = json.loads((folder / "tests_result.json").read_text())
    preservation = json.loads((folder / "input_preservation.json").read_text())
    if (coverage.get("status") != "complete_descriptive" or coverage.get("planned_units") != 96
            or coverage.get("recorded_units") != 96 or coverage.get("status_counts") != {"complete": 96}
            or coverage.get("inputs_unchanged_after_analysis") is not True):
        raise ReleaseError("Completed orthogonality coverage contract is inconsistent")
    if tests.get("tests") != 8 or tests.get("status") != "PASSED" or tests.get("exit_code") != 0 or tests.get("executed_before_real_bank_analysis") is not True:
        raise ReleaseError("Prior orthogonality test record is inconsistent")
    if preservation.get("unchanged") is not True or preservation.get("before") != preservation.get("after"):
        raise ReleaseError("Prior orthogonality input-preservation record is inconsistent")
    with (folder / "cell_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(row["cell"], row["dimension"]) for row in rows}
    if len(rows) != 96 or len(keys) != 96 or any(row["status"] != "complete" for row in rows):
        raise ReleaseError("Completed orthogonality table must retain all 96 unique conditions")
    for dimension in ("16", "8"):
        for domain in ("C7", "D4"):
            if sum(row["dimension"] == dimension and row["domain"] == domain for row in rows) != 24:
                raise ReleaseError("Completed orthogonality panel is incomplete")
    study = root / "studies/orthogonal_comparator_v1"
    source = json.loads((study / "source_manifest.json").read_text())
    for base, group in ((study, source["files"]), (root, source["project_dependencies"])):
        for name, expected in group.items():
            if sha256(relative_path(base, name)) != expected:
                raise ReleaseError("Orthogonality source snapshot changed: " + name)
    return {"previously_completed_units": 96, "previously_passed_numerical_tests": 8,
            "checked_imported_files": checked, "private_records_excluded_from_public_source": retained_private,
            "scientific_source_verified": True, "new_numerical_execution_during_verification": False}


def check_release(root: Path) -> dict:
    root = root.resolve()
    metadata_file = root / "release/release_manifest.json"
    if not metadata_file.is_file():
        raise ReleaseError("missing release/release_manifest.json")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    venue = metadata.get("target_venue", "ICML")
    if venue == "TMLR":
        if metadata.get("template_status") != "official_tmlr_unmodified":
            raise ReleaseError("TMLR release must use the unchanged official TMLR style")
        pending = metadata.get("pending_analyses", [])
        blockers = metadata.get("submission_readiness_blockers", [])
        if not isinstance(pending, list) or not isinstance(blockers, list):
            raise ReleaseError("pending analyses and submission blockers must be explicit lists")
        if type(metadata.get("submission_ready")) is not bool:
            raise ReleaseError("submission_ready must be an explicit boolean")
        if metadata["submission_ready"]:
            if pending or blockers or metadata.get("author_review_status") != "approved":
                raise ReleaseError("submission_ready=false is required until analyses and final author review are complete")
        elif not pending and not blockers:
            raise ReleaseError("a non-ready TMLR release must identify its remaining submission blocker")
    elif venue == "ICML":
        # Retain the old fixture/release contract for historical reproducibility.
        if metadata.get("target_year") != 2027 or metadata.get("template_year") != 2026:
            raise ReleaseError("release must distinguish target year 2027 from fallback template year 2026")
        if metadata.get("template_status") != "fallback_unverified_for_2027":
            raise ReleaseError("2027 template readiness must remain explicitly unverified")
    else:
        raise ReleaseError("unknown target venue")
    if metadata.get("paper_mode") != "anonymous" or metadata.get("numerical_pipeline_changed") is not False:
        raise ReleaseError("release must be anonymous with unchanged historical numerical pipeline")
    main = metadata.get("paper_main", "paper/main.tex")
    dependencies = tex_dependencies(root, main)
    text = uncomment_tex(relative_path(root, main).read_text(encoding="utf-8"))
    if venue == "TMLR" and ("\\usepackage{tmlr}" not in text or re.search(r"\\(?:icmltitle|icmlauthor|icmlfinalcopy|tmlrfinalcopy)\b", text)):
        raise ReleaseError("TMLR manuscript must use anonymous TMLR mode without ICML commands")
    if re.search(r"\\(?:usepackage|PassOptionsToPackage)\s*\[[^\]]*\baccepted\b", text):
        raise ReleaseError("accepted-mode manuscript is not anonymous submission mode")
    for name in re.findall(r"\\(?:icmlauthor|icmlcorrespondingauthor|author)\s*\{([^{}]*)\}", text):
        if name.strip() and "anonymous" not in name.lower():
            raise ReleaseError("nonanonymous author metadata in manuscript source")
    files = source_files(root)
    if not (root / "LICENSE").is_file():
        raise ReleaseError("missing preserved LICENSE")
    if metadata.get("original_license_sha256") != sha256(root / "LICENSE"):
        raise ReleaseError("original LICENSE checksum mismatch")
    scan_publication_files(root, files)
    prospective=None
    if metadata.get("prospective_pipeline_added"):
        try:prospective=__import__("verify_prospective").verify(root)
        except (ValueError,FileNotFoundError,KeyError) as exc:
            raise ReleaseError("prospective source manifest is missing or inconsistent: "+str(exc)) from exc
    characterization=None
    if metadata.get("current_study_manifest"):
        try: characterization=__import__("verify_characterization").verify(root)
        except (ValueError,FileNotFoundError,KeyError) as exc:
            raise ReleaseError("characterization source/workflow mismatch: "+str(exc)) from exc
    if metadata.get('reproduction_review'):
        try:
            __import__('build_reproduction_assets').load_review(root)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            raise ReleaseError('Reviewed replay evidence changed: '+str(exc)) from exc
    spectral = None
    if metadata.get("spectral_implementation_added"):
        try:
            spectral = __import__("verify_spectral").verify(root)
        except (ValueError, OSError, KeyError) as exc:
            raise ReleaseError("spectral source capsule changed: " + str(exc)) from exc
    context_generalization = None
    if metadata.get("context_generalization_implementation_added") or metadata.get("spectral_extensions_implementation_added"):
        try:
            context_generalization = __import__("verify_tmlr_studies").verify_all(root)
        except (ValueError, OSError, KeyError) as exc:
            raise ReleaseError("TMLR study source capsule changed: " + str(exc)) from exc
    orthogonal = check_orthogonal_results(root, metadata)
    return {"orthogonal_completed_results": orthogonal, "tmlr_study_sources": context_generalization, "spectral_source": spectral, "characterization_source": characterization, "prospective_source": prospective, "publication_files": len(files), "paper_dependencies": len(dependencies), "traced_artifacts": check_artifacts(root, dependencies), "frozen_source_files": check_source_manifest(root), "official_template_support_files": check_template(root), "reviewed_historical_inventory_files": check_review_coverage(root), "pdf_qa_current": check_pdf_qa(root, dependencies), "target_venue": venue, "target_year": metadata.get("target_year"), "template_year": metadata.get("template_year"), "submission_ready": metadata.get("submission_ready", False), "pending_analyses": metadata.get("pending_analyses", []), "submission_readiness_blockers": metadata.get("submission_readiness_blockers", []), "author_review_status": metadata.get("author_review_status"), "submission_ready_for_2027": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        print(json.dumps(check_release(args.root), indent=2, sort_keys=True))
    except (ReleaseError, KeyError, json.JSONDecodeError) as error:
        parser.exit(1, f"release check failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

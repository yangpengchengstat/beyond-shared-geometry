"""Release tests exercise missing evidence, portability and independent archive contents."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_release import ReleaseError, check_release, sha256, source_files, tex_dependencies
from package_release import package_release


def _fixture(root: Path) -> None:
    for directory in ("paper/figures", "release", "src/bsg", "configs", "provenance"):
        (root / directory).mkdir(parents=True)
    (root / "LICENSE").write_text("MIT License\nCopyright preserved\n")
    (root / "README.md").write_text("Portable source release.\n")
    (root / "src/bsg/__init__.py").write_text('VERSION = "frozen"\n')
    (root / "paper/main.tex").write_text("\\documentclass{article}\n\\usepackage{icml2026}\n\\begin{document}\n\\input{section}\n\\bibliographystyle{plainnat}\n\\bibliography{references}\n\\end{document}\n")
    (root / "paper/icml2026.sty").write_text("\\ProvidesPackage{icml2026}\n")
    (root / "paper/section.tex").write_text("\\citep{example}\n\\includegraphics{figures/evidence.pdf}\n")
    (root / "paper/references.bib").write_text("@article{example, title={Example}, author={Author}, year={2026}}\n")
    (root / "paper/figures/evidence.pdf").write_bytes(b"%PDF fixture evidence")
    metadata = {"target_year": 2027, "template_year": 2026, "template_status": "fallback_unverified_for_2027", "paper_main": "paper/main.tex", "paper_mode": "anonymous", "numerical_pipeline_changed": False, "original_license_sha256": sha256(root / "LICENSE")}
    (root / "release/release_manifest.json").write_text(json.dumps(metadata))
    artifact = {"path": "paper/figures/evidence.pdf", "sha256": sha256(root / "paper/figures/evidence.pdf"), "source_run": "fixture_frozen", "source_path": "results/figures/evidence.pdf", "role": "publication_figure"}
    (root / "release/artifact_manifest.json").write_text(json.dumps({"artifacts": [artifact]}))
    (root / "release/source_manifest.json").write_text(json.dumps({"files": {"src/bsg/__init__.py": sha256(root / "src/bsg/__init__.py")}}))
    (root / "provenance/template_source.json").write_text(json.dumps({"template_year": 2026, "modified": False, "support_sha256": {"icml2026.sty": sha256(root / "paper/icml2026.sty")}}))


def _rejects(operation, expected: str) -> None:
    try:
        operation()
    except ReleaseError as error:
        assert expected in str(error), str(error)
    else:
        raise AssertionError(f"release accepted invalid fixture: {expected}")


def test_release_checks_real_publication_contract():
    summary = check_release(ROOT)
    assert summary["traced_artifacts"] > 0
    assert summary["frozen_source_files"] > 0
    assert summary["submission_ready_for_2027"] is False


def test_release_follows_nested_tex_and_requires_local_style_and_citations():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); _fixture(root)
        closure = {str(path.relative_to(root)) for path in tex_dependencies(root)}
        assert {"paper/section.tex", "paper/references.bib", "paper/icml2026.sty", "paper/figures/evidence.pdf"} <= closure
        (root / "paper/icml2026.sty").unlink()
        _rejects(lambda: tex_dependencies(root), "missing usepackage")


def test_release_rejects_missing_assets_untraced_assets_and_modified_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); _fixture(root)
        (root / "paper/figures/evidence.pdf").write_bytes(b"changed result")
        _rejects(lambda: check_release(root), "artifact checksum mismatch")
        (root / "release/artifact_manifest.json").write_text('{"artifacts": []}')
        _rejects(lambda: check_release(root), "untraced publication artifacts")
        (root / "paper/figures/evidence.pdf").unlink()
        _rejects(lambda: check_release(root), "missing includegraphics")


def test_release_rejects_private_paths_credentials_and_changed_numerical_source():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); _fixture(root)
        (root / "README.md").write_text("/" + "Users/alice/research/input.csv")
        _rejects(lambda: check_release(root), "private absolute filesystem path")
        (root / "README.md").write_text("hf_" + "A" * 30)
        _rejects(lambda: check_release(root), "possible credential")
        (root / "README.md").write_text("portable")
        (root / "src/bsg/__init__.py").write_text('VERSION = "changed"\n')
        _rejects(lambda: check_release(root), "frozen numerical source changed")
        (root / "src/bsg/__init__.py").write_text('VERSION = "frozen"\n')
        (root / "paper/icml2026.sty").write_text("\\ProvidesPackage{icml2026}\n% changed margins\n")
        _rejects(lambda: check_release(root), "official template support file changed")


def test_release_rejects_tex_traversal_and_unresolved_citation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); _fixture(root)
        (root / "paper/section.tex").write_text("\\input{../private}\n")
        _rejects(lambda: tex_dependencies(root), "nonportable TeX dependency")
        (root / "paper/section.tex").write_text("\\citep{missing}\n")
        _rejects(lambda: tex_dependencies(root), "unresolved bibliography keys")


def test_release_rejects_stale_pdf_or_failed_saved_qa_without_pdf_runtime():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); _fixture(root)
        pdf = root / "paper/main.pdf"; pdf.write_bytes(b"%PDF compiled fixture")
        qa = {"pdf_path": "paper/main.pdf", "pdf_sha256": sha256(pdf), "source_sha256": {str(path.relative_to(root)): sha256(path) for path in tex_dependencies(root)}, "errors": []}
        qa_path = root / "release/pdf_qa.json"; qa_path.write_text(json.dumps(qa))
        assert check_release(root)["pdf_qa_current"] is True
        source = root / "paper/main.tex"; original = source.read_text(); source.write_text(original + "\n% manuscript edit\n")
        _rejects(lambda: check_release(root), "saved PDF is stale")
        source.write_text(original)
        pdf.write_bytes(b"%PDF replaced artifact")
        _rejects(lambda: check_release(root), "checksum does not match compiled PDF")
        pdf.write_bytes(b"%PDF compiled fixture")
        qa["errors"] = ["nonanonymous PDF metadata"]; qa_path.write_text(json.dumps(qa))
        _rejects(lambda: check_release(root), "saved PDF QA contains errors")


def test_release_archives_are_deterministic_self_contained_and_exclude_private_material():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "project"; root.mkdir(); _fixture(root)
        (root / ".git").mkdir(); (root / ".git/config").write_text("private history")
        (root / "src/bsg/__pycache__").mkdir(); (root / "src/bsg/__pycache__/private.pyc").write_bytes(b"cache")
        (root / "src/bsg/weights.safetensors").write_bytes(b"weights")
        one = package_release(root, Path(tmp) / "one")
        two = package_release(root, Path(tmp) / "two")
        assert one["archives"] == two["archives"]
        with zipfile.ZipFile(Path(tmp) / "one/overleaf.zip") as archive:
            assert {"main.tex", "section.tex", "icml2026.sty", "references.bib", "figures/evidence.pdf"} <= set(archive.namelist())
            assert not {"LICENSE", "README.md", "pyproject.toml"} & set(archive.namelist())
            extracted = Path(tmp) / "extracted"; archive.extractall(extracted)
        assert {path.name for path in tex_dependencies(extracted, "main.tex")} >= {"main.tex", "section.tex", "references.bib", "icml2026.sty"}
        with zipfile.ZipFile(Path(tmp) / "one/github-source.zip") as archive:
            names = set(archive.namelist())
            assert "beyond-shared-geometry/LICENSE" in names
            assert not any(".git/" in name or "__pycache__" in name or name.endswith(".safetensors") for name in names)

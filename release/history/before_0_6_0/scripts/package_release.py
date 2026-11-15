#!/usr/bin/env python3
"""Build deterministic source and anonymous Overleaf archives; never run models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from check_release import check_release, source_files, tex_dependencies

STAMP = (2026, 11, 7, 0, 0, 0)


def write_zip(path: Path, members: dict[str, bytes]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=STAMP)
            info.create_system = 3
            info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def package_release(root: Path, output_dir: Path) -> dict:
    root = root.resolve()
    summary = check_release(root)
    metadata = json.loads((root / "release/release_manifest.json").read_text(encoding="utf-8"))
    main = metadata.get("paper_main", "paper/main.tex")
    paper_root = (root / main).parent
    source_members = {"beyond-shared-geometry/" + str(path.relative_to(root)): path.read_bytes() for path in source_files(root)}
    paper_members = {str(path.relative_to(paper_root)): path.read_bytes() for path in tex_dependencies(root, main)}
    # Include saved source and hash receipt for caption-only derivatives. They
    # are needed by the portable formatter even when TeX uses only the derivative.
    for receipt in sorted((paper_root / "generated").glob("*.provenance.json")):
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if record.get("operation") != "caption_only_wording_revision_no_scientific_computation":
            continue
        output = record.get("output")
        if output not in paper_members:
            continue
        source = Path(record["source"])
        if source.is_absolute() or ".." in source.parts:
            raise ValueError("caption provenance source must stay within paper/")
        path = paper_root / source
        if path.is_symlink() or not path.is_file():
            raise ValueError("caption provenance source is missing or linked")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("source_sha256"):
            raise ValueError("caption provenance source checksum mismatch")
        if hashlib.sha256(paper_members[output]).hexdigest() != record.get("output_sha256"):
            raise ValueError("caption provenance output checksum mismatch")
        paper_members[source.as_posix()] = path.read_bytes()
        paper_members[str(receipt.relative_to(paper_root))] = receipt.read_bytes()
    # Supplementary full-precision endpoints and table-only generators are part
    # of the review source even when not referenced by a TeX input command.
    supplement = paper_root / "supplementary"
    if supplement.is_dir():
        for path in sorted(supplement.rglob("*")):
            if path.is_symlink():
                raise ValueError("manuscript supplement cannot contain symlinks")
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.suffix not in {".py", ".md", ".txt", ".csv", ".json"}:
                raise ValueError("unexpected manuscript supplement file type")
            paper_members[str(path.relative_to(paper_root))] = path.read_bytes()
    if (paper_root / "main.pdf").is_file():
        if (paper_root / "main.pdf").stat().st_size > 100_000_000:
            raise ValueError("compiled manuscript exceeds this release's 100 MB portability guard")
        paper_members["main.pdf"] = (paper_root / "main.pdf").read_bytes()
    template_license = paper_root / "TMLR_TEMPLATE_LICENSE.txt"
    if template_license.is_file():
        paper_members["TMLR_TEMPLATE_LICENSE.txt"] = template_license.read_bytes()
    venue = metadata.get("target_venue", "ICML")
    if venue == "TMLR":
        pending = metadata.get("pending_analyses", [])
        blockers = metadata.get("submission_readiness_blockers", [])
        ready = metadata["submission_ready"]
        if pending:
            status_text = "The listed analyses remain pending; no missing outcomes are represented as results.\n"
        else:
            status_text = "The five follow-up studies, saved 8/16 sensitivity and 96-unit independent-orthogonality comparison are completed evidence, integrated without rerunning them.\n"
        if not ready:
            status_text += "Submission readiness remains false. Remaining review: " + ", ".join(blockers or pending) + ".\n"
        paper_members["OVERLEAF_README.txt"] = (
            "Set main.tex as the main document. Compile with pdfLaTeX and BibTeX (or latexmk -pdf).\n"
            "This anonymous TMLR draft includes the unchanged official TMLR support files.\n"
            + status_text +
            "Saved-endpoint orthogonality and dimension-sensitivity generators are under supplementary/orthogonal_comparator/ and supplementary/spectral_dimension_sensitivity/. Current source labels define their manuscript locations.\n"
            "The public GitHub source archive contains author/license metadata and is not anonymous review material.\n"
        ).encode("utf-8")
        submission = {"target_venue": "TMLR", "template_status": "official_tmlr_unmodified",
                      "paper_mode": "anonymous", "main_document": "main.tex", "submission_ready": ready,
                      "pending_analyses": pending, "submission_readiness_blockers": blockers,
                      "author_review_status": metadata.get("author_review_status"),
                      "release_version": metadata.get("release_version")}
    else:
        paper_members["OVERLEAF_README.txt"] = (
            "Set the main document to main.tex and compile with pdfLaTeX and BibTeX (or latexmk -pdf).\n"
            "This historical archive uses the ICML 2026 fallback for the unverified 2027 target.\n"
        ).encode("utf-8")
        submission = {"target_venue": "ICML", "target_year": 2027, "template_year": 2026,
                      "template_status": "fallback_unverified_for_2027", "paper_mode": "anonymous",
                      "main_document": "main.tex", "submission_ready_for_2027": False}
    # Keep the active manuscript metadata synchronized with the portable copies.
    active_metadata = paper_root / "submission_metadata.json"
    if venue == "TMLR" and active_metadata.is_file():
        active = json.loads(active_metadata.read_text(encoding="utf-8"))
        for key in ("target_venue", "paper_mode", "main_document", "submission_ready", "release_version"):
            if active.get(key) != submission.get(key):
                raise ValueError("active manuscript metadata disagrees with release: " + key)
        submission.update(active)
        # Portable copies carry the local audit reports referenced by metadata.
        for key in ("audit_report", "revision_report", "dependency_audit", "compile_report", "style_report", "editorial_diff", "preservation_report"):
            value = active.get(key)
            if not value:
                continue
            report = (paper_root / value).resolve()
            if not report.is_file() or not report.is_relative_to(root):
                raise ValueError("active manuscript report is missing or external: " + key)
            portable = "review_notes/" + report.name
            paper_members[portable] = report.read_bytes()
            submission[key] = portable
    active_readme = paper_root / "OVERLEAF_README.txt"
    if venue == "TMLR" and active_readme.is_file():
        paper_members["OVERLEAF_README.txt"] = active_readme.read_bytes()
    paper_members["submission_metadata.json"] = json.dumps(submission, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    bundles = [("github-source.zip", source_members), ("overleaf.zip", paper_members)]
    if (root / "review_reproduction/manifest.json").is_file():
        from anonymous_review_package import anonymous_members
        bundles.append(("anonymous-review-source.zip", anonymous_members(root, paper_members)))
    for name, members in bundles:
        files[name] = {"sha256": write_zip(output_dir / name, members), "members": len(members)}
    report = {"archives": files, "validation": summary}
    (output_dir / "release_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-submission-ready", action="store_true", help="Refuse a release awaiting analyses or final author review")
    args = parser.parse_args(argv)
    if args.require_submission_ready:
        metadata = json.loads((args.root / "release/release_manifest.json").read_text())
        if (metadata.get("submission_ready") is not True or metadata.get("pending_analyses") or
                metadata.get("submission_readiness_blockers") or metadata.get("author_review_status") != "approved"):
            remaining = metadata.get("pending_analyses", []) + metadata.get("submission_readiness_blockers", [])
            parser.error("This is a preparation draft: submission readiness is false; remaining review: " +
                         ", ".join(remaining or ["final_author_review"]) + ".")
    print(json.dumps(package_release(args.root, args.output_dir or args.root / "dist"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

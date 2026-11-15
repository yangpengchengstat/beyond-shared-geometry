#!/usr/bin/env python3
"""Build deterministic source and anonymous Overleaf archives; never run models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from check_release import check_release, source_files, tex_dependencies

STAMP = (2026, 10, 25, 0, 0, 0)


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
    if (paper_root / "main.pdf").is_file():
        if (paper_root / "main.pdf").stat().st_size > 50_000_000:
            raise ValueError("compiled manuscript exceeds the 2026 fallback 50 MB limit")
        paper_members["main.pdf"] = (paper_root / "main.pdf").read_bytes()
    paper_members["OVERLEAF_README.txt"] = (
        "Set the main document to main.tex and compile with pdfLaTeX and BibTeX (or latexmk -pdf).\n"
        "This archive contains anonymous manuscript sources, bibliography, local style, and all referenced assets.\n"
        "Target: ICML 2027. The template is the ICML 2026 fallback; 2027 format and submission rules remain unverified.\n"
        "Compilation and bundle validation do not establish scientific acceptance or 2027 submission readiness.\n"
    ).encode("utf-8")
    paper_members["submission_metadata.json"] = json.dumps({
        "target_venue": "ICML", "target_year": 2027, "template_year": 2026,
        "template_status": "fallback_unverified_for_2027", "paper_mode": "anonymous",
        "main_document": "main.tex", "submission_ready_for_2027": False,
    }, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, members in (("github-source.zip", source_members), ("overleaf.zip", paper_members)):
        files[name] = {"sha256": write_zip(output_dir / name, members), "members": len(members)}
    report = {"archives": files, "validation": summary}
    (output_dir / "release_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(package_release(args.root, args.output_dir or args.root / "dist"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Assemble anonymous manuscript sources and a saved-endpoint review capsule."""
from pathlib import Path
import hashlib
import json
import re


def anonymous_members(root: Path, paper_members: dict[str, bytes]) -> dict[str, bytes]:
    capsule = root / "review_reproduction"
    manifest = json.loads((capsule / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe review-capsule path")
        path = capsule / rel
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Review-capsule checksum mismatch: " + name)
    files = {}
    for path in sorted(capsule.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError("Review capsule cannot include symlinks")
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".md", ".txt", ".csv", ".json"} and path.name != "LICENSE":
            raise ValueError("Unexpected review capsule member: " + path.name)
        files["reproduction/" + path.relative_to(capsule).as_posix()] = path.read_bytes()
    if "reproduction/reproduce.py" not in files:
        raise ValueError("Missing self-contained reproduction entrypoint")
    files.update({"paper/" + name: value for name, value in paper_members.items()})
    files["README.txt"] = (
        "Anonymous manuscript and saved-endpoint reproduction supplement\n\n"
        "Compile paper/main.tex, or open paper/main.pdf.\n"
        "From this directory, run:\n"
        "  python reproduction/reproduce.py --output /tmp/fresh-review-tables\n"
        "  python -m unittest discover -s reproduction/tests -v\n"
        "Python standard library only; choose a fresh output directory.\n"
        "This reproduces Tables 3-6 and the Section 5.1 heterogeneity summary from saved full-precision endpoints.\n"
        "Current core Table 1 has a table-only generator and full-precision inputs under paper/supplementary/orthogonal_comparator. The earlier core_risk_summary supplement is retained for provenance.\n"
        "Appendix K/Table 32 has a separate table-only generator and full-precision evidence under paper/supplementary/spectral_dimension_sensitivity.\n"
        "It does not repeat model execution, spectral analysis, coordinate residual checks, simulation or LOFO.\n"
        "Read reproduction/README.md and its input mapping before interpreting outputs.\n"
        "The separately supplied public GitHub release retains authorship and license metadata and is not anonymous.\n"
    ).encode()
    identifying = re.compile(r"Pengcheng|pengchengyang|LLM_PJ|/Users/|/home/|github\.com/[^\s/]+/beyond-shared-geometry", re.I)
    for name, raw in files.items():
        if Path(name).suffix != ".pdf" and identifying.search(raw.decode("utf-8")):
            raise ValueError("Identifying text in anonymous member: " + name)
    files["CONTENTS.sha256.json"] = (json.dumps(
        {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(files.items())},
        indent=2, sort_keys=True) + "\n").encode()
    return files

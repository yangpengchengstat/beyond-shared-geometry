#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/paper"
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs --keep-intermediates main.tex
else
  echo "Install TeX Live/latexmk or Tectonic, or upload dist/overleaf.zip to Overleaf."
  exit 1
fi

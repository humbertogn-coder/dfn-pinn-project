#!/usr/bin/env bash
# Build docs/PROJECT_REPORT.docx from docs/PROJECT_REPORT.md (pandoc).
# ```latex blocks become $$...$$ so that Word gets native equations.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp --suffix=.md)
python3 - "$tmp" <<'PY'
import re, sys
s = open("docs/PROJECT_REPORT.md", encoding="utf-8").read()
s = re.sub(r"```latex\n(.*?)\n```", lambda m: "$$" + m.group(1) + "$$", s, flags=re.S)
open(sys.argv[1], "w", encoding="utf-8").write(s)
PY
pandoc "$tmp" -o docs/PROJECT_REPORT.docx --resource-path=docs --toc
rm -f "$tmp"
echo "wrote docs/PROJECT_REPORT.docx"

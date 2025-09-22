#!/bin/bash
# Launch KRAFT training via uv
cd "$(dirname "$0")"
uv run kraft-train "$@"

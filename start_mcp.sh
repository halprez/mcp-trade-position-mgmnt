#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR"

# Try to find uv in common locations
if command -v uv >/dev/null 2>&1; then
    uv run python mcp_server.py
elif [ -f "$HOME/.local/bin/uv" ]; then
    "$HOME/.local/bin/uv" run python mcp_server.py
elif [ -f "/usr/local/bin/uv" ]; then
    /usr/local/bin/uv run python mcp_server.py
else
    echo "Error: uv not found. Please install uv or add it to PATH"
    exit 1
fi
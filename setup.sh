#!/usr/bin/env bash
set -e
PY=python3
VENV_DIR=.venv_color_adj

echo "Creating virtualenv in $VENV_DIR using $PY"
$PY -m venv "$VENV_DIR"

echo "Activating virtualenv and installing requirements"
# Use this activation command in interactive shells: source .venv_color_adj/bin/activate
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

echo "Done. To use the venv: source $VENV_DIR/bin/activate"

echo "If you have a GPU and want optimized installs, consider adding bitsandbytes and a compatible torch build." 

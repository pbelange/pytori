#!/bin/bash
set -euo pipefail; IFS=$'\n\t'

# Choose Python interpreter
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "❌ Neither python3 nor python found in PATH."
    exit 1
fi

echo "Using interpreter: $($PYTHON --version)"

#############################################
# Extract version from pytori/_version.py   #
#############################################

VERSION=$($PYTHON -c "import pytori._version as v; print(v.__version__)")

echo "Detected version: $VERSION"

NAME="pytori"

echo "========================================================================"
echo "Tagging $NAME v$VERSION"
echo "========================================================================"

git tag "v$VERSION"
git push origin "v$VERSION"

echo "========================================================================"
echo "Building $NAME v$VERSION"
echo "========================================================================"

$PYTHON -m build

echo "========================================================================"
echo "Uploading $NAME v$VERSION to PyPI"
echo "========================================================================"

# IF FIRST TIME ON PYPI USE:
# twine upload dist/* --verbose

twine upload dist/* --verbose --repository pytori

echo "Cleaning up build artifacts..."
rm -rf dist/ *.egg-info

echo "========================================================================"
echo "Release complete!"
echo "========================================================================"

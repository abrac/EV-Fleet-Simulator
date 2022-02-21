#!/bin/bash
python -m build
twine upload dist/* --skip-existing

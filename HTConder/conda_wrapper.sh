#!/bin/bash

source /home/$(whoami)/miniconda3/etc/profile.d/conda.sh
conda activate research

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

cd $SCRIPT_DIR
cd ..

python scripts/model_approaches/mulitple_models_v2.py
#!/usr/bin/env bash
set -e

pip install -r requirements.txt

# Pre-download the model weights during the BUILD step, not the first
# request. Without this, the first person to use the app after a deploy
# also pays for an ~80-200MB download on top of everything else — and on
# a slow/constrained instance that alone can cause the first request to
# time out or fail.
python -c "
import os
os.environ.setdefault('DEMUCS_MODEL', 'htdemucs')
from demucs.pretrained import get_model
get_model(os.environ['DEMUCS_MODEL'])
print('Model weights cached.')
"

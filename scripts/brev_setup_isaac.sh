#!/usr/bin/env bash
set -euo pipefail

echo "For Python/pip: use scripts/bootstrap_dev_env.sh (upgrades pip first; project needs Python 3.11+ and openreward from current PyPI)."
echo "If pip install -e . failed with 'No matching distribution' for openreward, your pip was too old; upgrade before install."
echo ""
echo "Run this on an RTX-capable Brev VM (L40S/L40/A10/A10G/L4)."
echo "It prepares the Isaac Sim container cache directories and prints the run command."

mkdir -p "$HOME/docker/isaac-sim/cache/kit" \
  "$HOME/docker/isaac-sim/cache/ov" \
  "$HOME/docker/isaac-sim/cache/pip" \
  "$HOME/docker/isaac-sim/cache/glcache" \
  "$HOME/docker/isaac-sim/cache/computecache" \
  "$HOME/docker/isaac-sim/logs" \
  "$HOME/docker/isaac-sim/data" \
  "$HOME/docker/isaac-sim/documents"

PUBLIC_IP="$(curl -s ifconfig.me || true)"

cat <<EOF
Pull Isaac Sim:
  docker pull nvcr.io/nvidia/isaac-sim:5.0.0

Start Isaac Sim shell with this repo mounted:
  docker run --name isaac-sim --entrypoint bash -it --runtime=nvidia --gpus all --rm --network=host \\
    -e ACCEPT_EULA=Y \\
    -e PRIVACY_CONSENT=Y \\
    -v "\$PWD":/workspace/complex_worlds_hack:rw \\
    -v "$HOME/docker/isaac-sim/cache/kit":/isaac-sim/kit/cache:rw \\
    -v "$HOME/docker/isaac-sim/cache/ov":/root/.cache/ov:rw \\
    -v "$HOME/docker/isaac-sim/cache/pip":/root/.cache/pip:rw \\
    -v "$HOME/docker/isaac-sim/cache/glcache":/root/.cache/nvidia/GLCache:rw \\
    -v "$HOME/docker/isaac-sim/cache/computecache":/root/.nv/ComputeCache:rw \\
    -v "$HOME/docker/isaac-sim/logs":/root/.nvidia-omniverse/logs:rw \\
    -v "$HOME/docker/isaac-sim/data":/root/.local/share/ov/data:rw \\
    -v "$HOME/docker/isaac-sim/documents":/root/Documents:rw \\
    nvcr.io/nvidia/isaac-sim:5.0.0

Inside the container:
  cd /workspace/complex_worlds_hack
  /isaac-sim/python.sh scripts/isaac/replay_breakfast_tray.py \\
    --baseline-trace artifacts/traces/baseline_fixed_script.jsonl \\
    --trained-trace artifacts/traces/dense_trained.jsonl \\
    --output-dir artifacts/isaac

For livestream, expose ports 49100 and 47998 in Brev, then run:
  PUBLIC_IP=${PUBLIC_IP} ./runheadless.sh --/app/livestream/publicEndpointAddress=${PUBLIC_IP} --/app/livestream/port=49100
EOF

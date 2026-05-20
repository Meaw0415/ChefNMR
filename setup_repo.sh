#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="${ROOT_DIR}/vendor"
CHEFNMR_DIR="${VENDOR_DIR}/chefnmr"

mkdir -p "${VENDOR_DIR}"

if [[ ! -d "${CHEFNMR_DIR}/.git" ]]; then
  git clone https://github.com/ml-struct-bio/chefnmr.git "${CHEFNMR_DIR}"
else
  echo "ChefNMR vendor repo already exists at ${CHEFNMR_DIR}"
fi

echo "Next steps:"
echo "  conda env create -f ${ROOT_DIR}/environment.yaml -n nmr3d"
echo "  export CHEFNMR_DIR=${CHEFNMR_DIR}"
echo "  export CHEFNMR_CKPT_DIR=${ROOT_DIR}/checkpoints"
echo "  mkdir -p ${ROOT_DIR}/checkpoints"
echo "  place datasets/checkpoints under the expected locations"

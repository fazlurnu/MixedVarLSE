#!/usr/bin/env bash
# Set up the TRC campaign stack on a server, from scratch or on top of a previous setup.
#
#   bash setup_server.sh                 # everything under ~/TRC
#   TRC_ROOT=/data/TRC bash setup_server.sh
#   bash setup_server.sh --skip-tests    # skip the two test suites (a few minutes)
#
# It clones (or fast-forwards) MixedVarLSENew and BlueSkyCDaRR on main, creates the conda
# env "mvlse" (Python 3.12) if missing, installs both packages editable -- BlueSkyCDaRR
# pulls the BlueSky fork at the commit its pyproject.toml pins -- registers the Jupyter
# kernel, runs the tests and prints the commits a campaign would record.
# Needs: git, conda (Miniconda/Anaconda) on PATH, network access to github.com.
set -euo pipefail

ROOT="${TRC_ROOT:-$HOME/TRC}"
ENV_NAME="mvlse"
PYTHON_VERSION="3.12"
RUN_TESTS=1
[[ "${1:-}" == "--skip-tests" ]] && RUN_TESTS=0

REPOS=(
  "MixedVarLSENew https://github.com/fazlurnu/MixedVarLSE.git"
  "BlueSkyCDaRR   https://github.com/fazlurnu/blueskycdarr.git"
)

say() { printf '\n==> %s\n' "$*"; }

command -v git >/dev/null || { echo "git not found"; exit 1; }
command -v conda >/dev/null || { echo "conda not found: install Miniconda first"; exit 1; }

# ------------------------------------------------------------------ repositories
say "Repositories under $ROOT"
mkdir -p "$ROOT"
for entry in "${REPOS[@]}"; do
  read -r name url <<<"$entry"
  dir="$ROOT/$name"
  if [[ -d "$dir/.git" ]]; then
    if [[ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]]; then
      echo "$name has uncommitted changes; leaving it as it is"
    else
      git -C "$dir" fetch --quiet origin
      git -C "$dir" checkout --quiet main
      git -C "$dir" merge --ff-only --quiet origin/main
      echo "$name updated to $(git -C "$dir" rev-parse --short HEAD)"
    fi
  else
    git clone --quiet --branch main "$url" "$dir"
    echo "$name cloned at $(git -C "$dir" rev-parse --short HEAD)"
  fi
done

# ------------------------------------------------------------------ environment
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  say "Conda env $ENV_NAME exists"
else
  say "Creating conda env $ENV_NAME (Python $PYTHON_VERSION)"
  conda create --yes --quiet --name "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"
python -m pip install --quiet --upgrade pip

say "Installing BlueSkyCDaRR (pulls the pinned BlueSky fork)"
python -m pip install --quiet -e "$ROOT/BlueSkyCDaRR"

say "Installing mvlse and the notebook tooling"
python -m pip install --quiet -e "$ROOT/MixedVarLSENew[plots,tables]"
python -m pip install --quiet joblib pyyaml pytest jupyter nbconvert ipykernel
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)" >/dev/null

# ------------------------------------------------------------------ checks
if [[ "$RUN_TESTS" == 1 ]]; then
  say "Testing BlueSkyCDaRR"
  (cd "$ROOT/BlueSkyCDaRR" && python -m pytest -q -p no:warnings)
  say "Testing mvlse"
  (cd "$ROOT/MixedVarLSENew" && python -m pytest -q -p no:warnings)
fi

say "What a campaign would record (both must say dirty: False)"
(cd "$ROOT/MixedVarLSENew/paper_trc" && python - <<'EOF'
import common, config, os
for name, s in common.code_state().items():
    print(f"  {name:13s} {s['commit'][:7]}  dirty: {s['dirty']}")
print(f"  N_JOBS in config.py: {config.N_JOBS}; this machine has {os.cpu_count()} cores")
EOF
)

cat <<EOF

Done. Next:
  conda activate $ENV_NAME
  cd $ROOT/MixedVarLSENew/paper_trc
  # raise N_JOBS in config.py to suit this machine (memory: one BlueSky per worker),
  # commit and push -- results do not depend on it, but a run refuses dirty code
  tmux new -s trc
  python exp2_surface.py --dpsi 180 --bracket
EOF

#!/usr/bin/env bash
#
# setup_env.sh -- build the virtual environment that synthetic_fraud_validation.ipynb
#                 needs, then prove it is complete before you open the notebook.
#
# The notebook is bit-for-bit reproducible only on the pinned versions in
# requirements.txt (see README.md). This script installs exactly those, registers a
# Jupyter kernel pointing at them, and runs a preflight check covering every import
# the notebook makes, the input data file, and the output directories.
#
# Usage:
#   ./setup_env.sh                 create .venv and install everything
#   ./setup_env.sh --cpu-torch     same, but pull the CPU-only torch wheel (~200 MB
#                                  instead of ~2.5 GB). The notebook runs CTGAN with
#                                  enable_gpu=False, so this changes nothing but size.
#   ./setup_env.sh --verify-only   skip installing, just re-run the preflight check
#   ./setup_env.sh --recreate      delete an existing .venv and start clean
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
REQ="$HERE/requirements.txt"
KERNEL_NAME="dat610"
KERNEL_LABEL="Python (DAT 610 synthetic fraud)"
PY_WANTED="3.12"

CPU_TORCH=0
VERIFY_ONLY=0
RECREATE=0
for arg in "$@"; do
  case "$arg" in
    --cpu-torch)   CPU_TORCH=1 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    --recreate)    RECREATE=1 ;;
    -h|--help)     sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[[ -f "$REQ" ]] || fail "requirements.txt not found next to this script."

# ---------------------------------------------------------------------------
# 1. Pick an interpreter. The pins were resolved against Python 3.12; a different
#    minor version can silently resolve to different wheels.
# ---------------------------------------------------------------------------
if [[ $VERIFY_ONLY -eq 0 ]]; then
  BASE_PY=""
  for cand in "python$PY_WANTED" python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then BASE_PY="$(command -v "$cand")"; break; fi
  done
  [[ -n "$BASE_PY" ]] || fail "no python interpreter found on PATH."

  FOUND_VER="$("$BASE_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  echo "interpreter : $BASE_PY (Python $FOUND_VER)"
  if [[ "$FOUND_VER" != "$PY_WANTED" ]]; then
    printf '\033[33mWARNING: requirements.txt was pinned on Python %s, you have %s.\n' \
           "$PY_WANTED" "$FOUND_VER"
    printf 'Install may still succeed, but the notebook numbers are only guaranteed on %s.\033[0m\n' \
           "$PY_WANTED"
  fi

  # -------------------------------------------------------------------------
  # 2. Create the venv.
  # -------------------------------------------------------------------------
  if [[ $RECREATE -eq 1 && -d "$VENV" ]]; then
    say "Removing existing $VENV"
    rm -rf "$VENV"
  fi
  if [[ -d "$VENV" ]]; then
    say "Reusing existing virtual environment at $VENV"
  else
    say "Creating virtual environment at $VENV"
    "$BASE_PY" -m venv "$VENV" || fail "venv creation failed. On Debian/Ubuntu you may need: apt install python3-venv"
  fi
fi

[[ -x "$VENV/bin/python" ]] || fail "no interpreter at $VENV/bin/python -- run without --verify-only first."
VPY="$VENV/bin/python"

if [[ $VERIFY_ONLY -eq 0 ]]; then
  # -------------------------------------------------------------------------
  # 3. Install. torch first when a special index is requested, so the pinned
  #    CPU wheel wins before ctgan pulls torch from the default index.
  # -------------------------------------------------------------------------
  say "Upgrading pip toolchain"
  "$VPY" -m pip install --quiet --upgrade pip setuptools wheel

  if [[ $CPU_TORCH -eq 1 ]]; then
    say "Installing CPU-only torch"
    TORCH_PIN="$(grep -E '^torch==' "$REQ")"
    "$VPY" -m pip install --index-url https://download.pytorch.org/whl/cpu "$TORCH_PIN" \
      || fail "CPU torch install failed. Retry without --cpu-torch."
  fi

  say "Installing pinned requirements (this takes a few minutes)"
  "$VPY" -m pip install -r "$REQ" || fail "dependency install failed -- see pip output above."

  say "Checking for dependency conflicts"
  "$VPY" -m pip check || printf '\033[33mpip reported conflicts above; the preflight check below is what matters.\033[0m\n'

  # -------------------------------------------------------------------------
  # 4. Register a Jupyter kernel so the notebook can be pointed at this venv.
  # -------------------------------------------------------------------------
  say "Registering Jupyter kernel '$KERNEL_NAME'"
  "$VPY" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_LABEL" >/dev/null
fi

# ---------------------------------------------------------------------------
# 5. Preflight. Every import the notebook makes, the data file, the output dirs,
#    and a real CTGAN fit on a tiny frame so a broken torch/ctgan pairing surfaces
#    here rather than 40 minutes into a training run.
# ---------------------------------------------------------------------------
say "Preflight check"
"$VPY" - "$HERE" <<'PYCHECK'
import importlib, os, sys

here = sys.argv[1]
ok = True

def bad(msg):
    global ok
    ok = False
    print(f"  \033[31mFAIL\033[0m {msg}")

def good(msg):
    print(f"  \033[32mok\033[0m   {msg}")

print(f"\npython {sys.version.split()[0]}  ({sys.executable})\n")

# Import name -> distribution name, for the pin comparison.
MODULES = [
    ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
    ("sklearn", "scikit-learn"), ("torch", "torch"), ("ctgan", "ctgan"),
    ("rdt", "rdt"), ("tqdm", "tqdm"),
    ("matplotlib", "matplotlib"), ("seaborn", "seaborn"), ("PIL", "pillow"),
    ("nbformat", "nbformat"), ("ipykernel", "ipykernel"), ("nbconvert", "nbconvert"),
]

# Expected versions straight from requirements.txt, so the two files cannot drift.
expected = {}
with open(os.path.join(here, "requirements.txt")) as fh:
    for line in fh:
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, ver = line.split("==", 1)
            expected[name.strip().lower()] = ver.strip()

print("packages:")
for mod, dist in MODULES:
    try:
        m = importlib.import_module(mod)
    except Exception as exc:
        bad(f"{mod}: cannot import ({type(exc).__name__}: {exc})")
        continue
    got = getattr(m, "__version__", "?")
    want = expected.get(dist.lower())
    # torch reports a local build tag (2.13.0+cu130); compare the release part only.
    base = got.split("+")[0]
    if want and base != want:
        print(f"  \033[33mwarn\033[0m {dist} {got} installed, requirements.txt pins {want}"
              f" -- numbers may differ from the recorded run")
    else:
        good(f"{dist} {got}")

# The specific symbols the notebook pulls in, not just the top-level packages.
print("\nnotebook imports:")
SYMBOLS = [
    ("scipy.stats", ["ks_2samp", "chi2_contingency", "wasserstein_distance"]),
    ("sklearn.model_selection", ["train_test_split"]),
    ("sklearn.ensemble", ["RandomForestClassifier"]),
    ("sklearn.preprocessing", ["OrdinalEncoder", "OneHotEncoder", "MinMaxScaler"]),
    ("sklearn.neighbors", ["NearestNeighbors"]),
    ("sklearn.metrics", ["roc_auc_score", "confusion_matrix", "classification_report",
                         "precision_score", "recall_score", "f1_score"]),
    ("ctgan", ["CTGAN"]),
]
for mod, names in SYMBOLS:
    try:
        m = importlib.import_module(mod)
        missing = [n for n in names if not hasattr(m, n)]
        if missing:
            bad(f"{mod}: missing {', '.join(missing)}")
        else:
            good(f"{mod} ({len(names)} symbol{'s' if len(names) > 1 else ''})")
    except Exception as exc:
        bad(f"{mod}: {type(exc).__name__}: {exc}")

# The two CTGAN calls the notebook depends on. set_random_state and enable_gpu are
# version-sensitive: older ctgan used `cuda=` and had no set_random_state.
print("\nctgan API:")
try:
    from ctgan import CTGAN
    import inspect
    sig = inspect.signature(CTGAN.__init__)
    if "enable_gpu" in sig.parameters:
        good("CTGAN(enable_gpu=...) accepted")
    else:
        bad("CTGAN has no enable_gpu parameter (ctgan too old -- notebook passes it)")
    if hasattr(CTGAN, "set_random_state"):
        good("CTGAN.set_random_state present")
    else:
        bad("CTGAN.set_random_state missing (ctgan too old -- seeding would silently no-op)")
except Exception as exc:
    bad(f"ctgan API check: {type(exc).__name__}: {exc}")

# Thread pinning is load-bearing for reproducibility, not a performance tweak.
print("\nreproducibility prerequisites:")
try:
    import torch
    torch.set_num_threads(4)
    good(f"torch.set_num_threads(4) accepted (threads now {torch.get_num_threads()})")
except Exception as exc:
    bad(f"torch.set_num_threads: {type(exc).__name__}: {exc}")

# Headless plotting: no display in most environments, so Agg must be reachable.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(); plt.close(fig)
    good("matplotlib Agg backend renders")
except Exception as exc:
    bad(f"matplotlib: {type(exc).__name__}: {exc}")

# Files and directories the notebook assumes exist.
print("\nfiles and directories:")
data_csv = os.path.join(here, "data", "carclaims.csv")
if os.path.isfile(data_csv):
    good(f"data/carclaims.csv ({os.path.getsize(data_csv) / 1e6:.1f} MB)")
else:
    bad("data/carclaims.csv not found -- the notebook reads this in its first cell. "
        "Download it from the Kaggle link in README.md and place it at data/carclaims.csv")

for d in ("figures", "outputs"):
    p = os.path.join(here, d)
    os.makedirs(p, exist_ok=True)
    if os.access(p, os.W_OK):
        good(f"{d}/ writable")
    else:
        bad(f"{d}/ is not writable")

# A real end-to-end CTGAN fit, kept tiny. Catches a broken torch/ctgan pairing in
# seconds instead of after the 300-epoch training cell.
print("\nsmoke test:")
try:
    import pandas as pd, numpy as np
    from ctgan import CTGAN
    rng = np.random.default_rng(0)
    toy = pd.DataFrame({
        "num": rng.normal(size=60),
        "cat": rng.choice(["a", "b", "c"], size=60),
    })
    model = CTGAN(epochs=1, batch_size=20, verbose=False, enable_gpu=False)
    model.set_random_state(42)
    model.fit(toy, discrete_columns=["cat"])
    sample = model.sample(10)
    if len(sample) == 10 and set(sample.columns) == {"num", "cat"}:
        good("CTGAN fit + sample on a toy frame")
    else:
        bad(f"CTGAN sampled unexpected shape: {sample.shape}, columns {list(sample.columns)}")
except Exception as exc:
    bad(f"CTGAN smoke test: {type(exc).__name__}: {exc}")

print()
if ok:
    print("\033[32mPREFLIGHT PASSED -- the notebook has everything it needs.\033[0m")
else:
    print("\033[31mPREFLIGHT FAILED -- fix the items above before running the notebook.\033[0m")
sys.exit(0 if ok else 1)
PYCHECK

say "Done"
cat <<EOF
Activate the environment:
    source .venv/bin/activate

Then either open the notebook and select the kernel "$KERNEL_LABEL":
    jupyter notebook synthetic_fraud_validation.ipynb

or run it start to finish without opening it (~40 min, CTGAN trains 300 epochs):
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace \\
        --ExecutePreprocessor.timeout=-1 synthetic_fraud_validation.ipynb

Re-check the environment at any time:
    ./setup_env.sh --verify-only
EOF

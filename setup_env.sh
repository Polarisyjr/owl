#!/bin/bash
# Set up the `owl` conda env + system deps needed to ACTUALLY run GAIA web-browsing
# tasks on this machine (Azure Linux 3.0 — no apt, so `playwright install-deps`
# does not work and several deps must be installed explicitly).
#
# What it does (all idempotent — safe to re-run):
#   1. ensure miniconda + conda env `owl` (python 3.11)
#   2. pip install -r requirements.txt  + pinned fixups upstream's file gets wrong
#   3. install the chromium browser binary for playwright
#   4. install chromium's system shared libraries (.so) via dnf
#   5. install fonts (Latin + CJK + emoji) so pages render glyphs, not tofu boxes
#   6. install the ffmpeg CLI (absent from Azure Linux repos) via conda-forge
#   7. verify: chromium launches headless + ffmpeg is on PATH
#
# Requires passwordless sudo for the dnf steps (4 & 5).
#
# Usage (from anywhere):
#   bash frameworks/owl/setup_env.sh             # full setup
#   bash frameworks/owl/setup_env.sh --verify    # skip installs, just run the checks
#
# Env overrides:
#   CONDA_HOME   conda install prefix   (default: $HOME/miniconda3)
#   CONDA_ENV    env name               (default: owl)

set -euo pipefail

CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-owl}"
OWL_DIR="$(cd "$(dirname "$0")" && pwd)"     # this script lives in frameworks/owl
ROOT="$(cd "$OWL_DIR/../.." && pwd)"
REQ="$OWL_DIR/requirements.txt"
VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=1

# chromium runtime .so deps (resolved from `dnf provides` on Azure Linux 3.0).
# playwright install-deps targets apt/Ubuntu and is a no-op here, so install by hand.
SYS_LIBS=(alsa-lib at-spi2-atk at-spi2-core mesa-libgbm nspr nss-libs nss)

# fonts: DejaVu (Latin), Noto CJK (中文/日本語/한국어), Noto emoji
FONT_PKGS=(fontconfig dejavu-sans-fonts dejavu-serif-fonts \
           google-noto-sans-cjk-ttc-fonts google-noto-emoji-fonts)

# pinned fixups NOT correctly captured by requirements.txt:
#   sqlalchemy        — camel storage imports it; missing from requirements
#   scenedetect 0.6.2 — newer drops VideoManager that vendored camel still uses
#   mcp 1.12.4        — newest mcp that still accepts pydantic<2.10 (>=1.13 needs 2.11)
#   pydantic <2.10    — required by requirements.txt + unstructured-client
PIN_FIXUPS=(sqlalchemy 'scenedetect==0.6.2' 'mcp==1.12.4' 'pydantic>=2.9.2,<2.10')

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m %s\n' "$*"; }
die()  { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Azure Linux runs a background tdnf (e.g. clamav). Don't fight the rpm lock —
# wait for it, never kill a transaction mid-flight.
wait_rpm_lock() {
    for _ in $(seq 1 180); do
        sudo fuser /var/lib/rpm/.rpm.lock >/dev/null 2>&1 || return 0
        sleep 5
    done
    die "rpm lock still held after 15min — another package manager is stuck"
}

# --no-capture-output so child stdout (verify prints, pip progress) streams through
conda_run() { "$CONDA_HOME/bin/conda" run --no-capture-output -n "$CONDA_ENV" "$@"; }

# ---------------------------------------------------------------------------
if [ "$VERIFY_ONLY" = "0" ]; then
    [ -f "$REQ" ] || die "requirements.txt not found at $REQ (init the owl submodule first)"

    say "1. conda env '$CONDA_ENV'"
    if [ ! -x "$CONDA_HOME/bin/conda" ]; then
        die "conda not found at $CONDA_HOME — install miniconda there, or set CONDA_HOME"
    fi
    if ! "$CONDA_HOME/bin/conda" env list | grep -qE "^${CONDA_ENV}\s"; then
        "$CONDA_HOME/bin/conda" create -y -n "$CONDA_ENV" python=3.11
    fi
    ok "env ready ($(conda_run python --version 2>&1))"

    say "2. python deps"
    conda_run pip install -r "$REQ"
    conda_run pip install "${PIN_FIXUPS[@]}"
    conda_run pip check || true   # report-only; harmless extras may remain
    ok "requirements + pinned fixups installed"

    say "3. playwright chromium browser binary"
    conda_run python -m playwright install chromium
    ok "chromium downloaded"

    say "4. chromium system libraries (dnf)"
    wait_rpm_lock
    sudo dnf install -y "${SYS_LIBS[@]}"
    ok "system libs installed"

    say "5. fonts"
    wait_rpm_lock
    sudo dnf install -y "${FONT_PKGS[@]}"
    sudo fc-cache -f >/dev/null 2>&1 || true
    ok "fonts installed ($(fc-list 2>/dev/null | wc -l) families)"

    say "6. ffmpeg CLI (conda-forge — not in Azure Linux repos)"
    if ! conda_run bash -c 'command -v ffmpeg' >/dev/null 2>&1; then
        "$CONDA_HOME/bin/conda" install -n "$CONDA_ENV" -c conda-forge ffmpeg -y
    fi
    ok "ffmpeg: $(conda_run bash -c 'command -v ffmpeg')"
fi

# ---------------------------------------------------------------------------
say "verify"

# any chromium .so still missing?
CHROME="$(ls -d "$HOME"/.cache/ms-playwright/chromium-*/chrome-linux64/chrome 2>/dev/null | head -1 || true)"
[ -n "$CHROME" ] || die "chromium binary not found — run without --verify first"
MISSING="$(ldd "$CHROME" 2>/dev/null | grep -i 'not found' | awk '{print $1}' | sort -u || true)"
if [ -n "$MISSING" ]; then
    echo "$MISSING" | sed 's/^/  MISSING /'
    die "chromium still has unresolved libraries (map them with: dnf provides '*/<lib>')"
fi
ok "no missing chromium libraries"

# real headless launch + glyph render
conda_run python - <<'PY'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page()
    pg.set_content("<h1>owl browser 中文 😀</h1>")
    txt = pg.evaluate("() => document.querySelector('h1').textContent")
    b.close()
    assert txt == "owl browser 中文 😀", txt
    print("  ok chromium launched headless and rendered:", repr(txt))
PY

# ffmpeg on PATH inside the env
conda_run bash -c 'command -v ffmpeg >/dev/null && echo "  ok ffmpeg: $(ffmpeg -version 2>/dev/null | head -1)"' \
    || die "ffmpeg not on PATH in env $CONDA_ENV"

printf '\n\033[1;32mowl environment ready.\033[0m  Run a task with: bash %s/scripts/owl/start.sh\n' "$ROOT"

#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/install_demucs_uv.sh [--python PATH]

Description:
  Install demucs into an existing uv-managed Python environment without
  allowing demucs to downgrade torchaudio.

Options:
  --python PATH   Target Python interpreter. If omitted, uv's active/default
                  environment is used.
  -h, --help      Show this help message.
EOF
}

PYTHON_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            if [[ $# -lt 2 ]]; then
                echo "error: --python requires a path" >&2
                exit 2
            fi
            PYTHON_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is not installed or not on PATH" >&2
    exit 1
fi

UV_PYTHON_ARGS=()
if [[ -n "$PYTHON_PATH" ]]; then
    UV_PYTHON_ARGS+=(--python "$PYTHON_PATH")
fi

uv_pip() {
    local subcommand="$1"
    shift
    uv pip "$subcommand" "${UV_PYTHON_ARGS[@]}" "$@"
}

echo "==> Checking existing ASR environment"
uv_pip show whisperx >/dev/null 2>&1 || {
    echo "error: whisperx is not installed in the target environment" >&2
    echo "hint: install whisperx first, then rerun this script" >&2
    exit 1
}

before_torchaudio="$(uv_pip freeze | awk -F '==' '$1 == "torchaudio" { print $2 }')"
before_torch="$(uv_pip freeze | awk -F '==' '$1 == "torch" { print $2 }')"

if [[ -z "$before_torch" || -z "$before_torchaudio" ]]; then
    echo "error: torch and torchaudio must already be installed in the target environment" >&2
    exit 1
fi

echo "    torch==$before_torch"
echo "    torchaudio==$before_torchaudio"

echo "==> Installing demucs without dependencies"
uv_pip install --no-deps "demucs[dev] @ git+https://github.com/adefossez/demucs"

echo "==> Installing demucs runtime dependencies"
uv_pip install dora-search openunmix lameenc

after_torchaudio="$(uv_pip freeze | awk -F '==' '$1 == "torchaudio" { print $2 }')"
after_torch="$(uv_pip freeze | awk -F '==' '$1 == "torch" { print $2 }')"

echo "==> Installed package versions"
echo "    torch==$after_torch"
echo "    torchaudio==$after_torchaudio"
uv_pip freeze | awk '
    BEGIN { FS = "==" }
    $1 == "demucs" || $1 == "whisperx" || $1 == "openunmix" || $1 == "dora-search" || $1 == "lameenc" {
        printf "    %s==%s\n", $1, $2
    }
'

if [[ "$before_torch" != "$after_torch" || "$before_torchaudio" != "$after_torchaudio" ]]; then
    echo "error: torch/torchaudio changed during install; investigate the environment before running ASR" >&2
    exit 1
fi

echo "==> Done"
echo "demucs was installed without modifying torch/torchaudio."

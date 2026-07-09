#!/usr/bin/env bash
# Shared R4 runtime: activate myenv310 so dataset bytecode (cpython-310) loads correctly.
ENV_NAME="${ENV_NAME:-myenv310}"

_r4_activate_conda() {
  local CONDA_SH=""
  for _c in \
    "${CONDA_BASE:-}/etc/profile.d/conda.sh" \
    "/usr/local/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    [[ -f "$_c" ]] && CONDA_SH="$_c" && break
  done
  if [[ -z "$CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
    local _b
    _b="$(conda info --base 2>/dev/null || true)"
    [[ -n "$_b" && -f "$_b/etc/profile.d/conda.sh" ]] && CONDA_SH="$_b/etc/profile.d/conda.sh"
  fi
  [[ -n "$CONDA_SH" ]] || return 1
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$ENV_NAME"
}

if [[ -z "${R4_PYTHON:-}" ]]; then
  if _r4_activate_conda && [[ -x "${CONDA_PREFIX}/bin/python3" ]]; then
    R4_PYTHON="${CONDA_PREFIX}/bin/python3"
  elif [[ -x "/home/lizhichun_24/.conda/envs/myenv310/bin/python3" ]]; then
    R4_PYTHON="/home/lizhichun_24/.conda/envs/myenv310/bin/python3"
  else
    R4_PYTHON="$(command -v python3)"
  fi
  export R4_PYTHON
fi

python3() { "$R4_PYTHON" "$@"; }
export -f python3 2>/dev/null || true

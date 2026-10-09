# Shared environment for the LIBERO + SmolVLA walkthrough.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "缺少虚拟环境。请先运行: bash scripts/setup.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

# Clash and similar clients export ALL_PROXY=socks://127.0.0.1:7890.
# httpx only accepts socks5://, and then only if socksio is installed.
normalize_proxy_scheme() {
  local value="$1"
  case "$value" in
    socks://*) printf 'socks5://%s' "${value#socks://}" ;;
    *) printf '%s' "$value" ;;
  esac
}
for _proxy_name in ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy SOCKS_PROXY socks_proxy; do
  if [[ -n "${!_proxy_name-}" ]]; then
    printf -v "$_proxy_name" '%s' "$(normalize_proxy_scheme "${!_proxy_name}")"
    export "$_proxy_name"
  fi
done
unset _proxy_name

# Headless MuJoCo on this machine (no display required).
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false

# LIBERO asks an interactive question the first time it is imported.
# Write the default config first so eval can run without a terminal prompt.
LIBERO_PKG="$ROOT/.venv/lib/python3.12/site-packages/libero/libero"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
if [[ ! -f "$LIBERO_CONFIG_DIR/config.yaml" && -d "$LIBERO_PKG" ]]; then
  mkdir -p "$LIBERO_CONFIG_DIR"
  python - "$LIBERO_PKG" "$LIBERO_CONFIG_DIR/config.yaml" <<'PY'
import sys
from pathlib import Path
import yaml

root = Path(sys.argv[1]).resolve()
config = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "datasets": str(root.parent / "datasets"),
    "assets": str(root / "assets"),
}
Path(sys.argv[2]).write_text(yaml.safe_dump(config), encoding="utf-8")
print(f"已写入 LIBERO 配置 {sys.argv[2]}")
PY
fi

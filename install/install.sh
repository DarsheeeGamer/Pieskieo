#!/usr/bin/env bash
set -euo pipefail

# Pieskieo installer (Linux/macOS)
# Linux: requires sudo/root; installs to /usr/local/bin and sets up systemd service pieskieo.
# macOS: installs to /usr/local/bin or ~/.local/bin (no service).

usage() {
  cat <<'EOF'
Pieskieo installer

Options:
  -p, --prefix DIR   Installation prefix (default: /usr/local if writable, else ~/.local)
  -h, --help         Show this help
EOF
}

PREFIX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--prefix) PREFIX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$(uname -s)" == "Linux" && "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "[installer] please run with sudo/root on Linux to install as a service"
  exit 1
fi

if [[ -z "${PREFIX}" ]]; then
  if [[ -w /usr/local/bin ]]; then
    PREFIX="/usr/local"
  else
    PREFIX="$HOME/.local"
    mkdir -p "$PREFIX/bin"
  fi
fi

echo "Installing to prefix: $PREFIX"

if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust toolchain (cargo) is required. Install via https://rustup.rs/ then rerun." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"
echo "Building Pieskieo in release mode..."
cargo build --release --locked

BIN_SRC="${REPO_ROOT}/target/release"
BIN_DST="${PREFIX}/bin"
mkdir -p "$BIN_DST"

for bin in pieskieo-server pieskieo load bench; do
  if [[ -f "${BIN_SRC}/${bin}" ]]; then
    install -m 0755 "${BIN_SRC}/${bin}" "${BIN_DST}/${bin}"
    echo "Installed ${bin} -> ${BIN_DST}/${bin}"
  fi
done

if [[ "$(uname -s)" == "Linux" ]]; then
  echo "[installer] configuring systemd service pieskieo"
  mkdir -p /var/lib/pieskieo
  cat >/etc/systemd/system/pieskieo.service <<'EOF'
[Unit]
Description=Pieskieo Database Service
After=network.target

[Service]
Type=simple
Environment=PIESKIEO_DATA=/var/lib/pieskieo
Environment=PIESKIEO_LISTEN=0.0.0.0:8000
ExecStart=/usr/local/bin/pieskieo-server --serve
Restart=on-failure
User=root
Group=root

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable pieskieo.service
  systemctl restart pieskieo.service
  echo "[installer] service pieskieo enabled and started"
fi

echo "Done. Ensure ${BIN_DST} is on your PATH."

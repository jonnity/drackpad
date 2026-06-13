#!/usr/bin/env bash
#
# Downloads the runtime dependencies that are too large / platform-specific to
# commit: the ONNX Runtime shared library and the hand-landmark model.
#
# After running, follow the printed instructions to set ORT_DYLIB_PATH and run
# the spike with `cargo run`.

set -euo pipefail

ORT_VERSION="1.20.1"
MODEL_URL="https://raw.githubusercontent.com/PINTO0309/hand-gesture-recognition-using-onnx/main/model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor"
MODEL_DIR="$REPO_ROOT/models"
MODEL_PATH="$MODEL_DIR/hand_landmark_sparse_Nx3x224x224.onnx"

# --- ONNX Runtime --------------------------------------------------------------
# `ort` is built with the `load-dynamic` feature, so the runtime is loaded at
# program start from ORT_DYLIB_PATH. The API level pinned in Cargo.toml
# (`api-20`) must match this version's minor number.
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  ORT_PKG="onnxruntime-linux-x64-${ORT_VERSION}" ;;
  Linux-aarch64) ORT_PKG="onnxruntime-linux-aarch64-${ORT_VERSION}" ;;
  Darwin-arm64)  ORT_PKG="onnxruntime-osx-arm64-${ORT_VERSION}" ;;
  Darwin-x86_64) ORT_PKG="onnxruntime-osx-x86_64-${ORT_VERSION}" ;;
  *) echo "Unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

ORT_DIR="$VENDOR_DIR/$ORT_PKG"
if [ ! -d "$ORT_DIR" ]; then
  echo "Downloading ONNX Runtime ${ORT_VERSION} ..."
  mkdir -p "$VENDOR_DIR"
  curl -fSL --retry 4 --retry-delay 2 \
    -o "$VENDOR_DIR/$ORT_PKG.tgz" \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_PKG}.tgz"
  tar xzf "$VENDOR_DIR/$ORT_PKG.tgz" -C "$VENDOR_DIR"
  rm -f "$VENDOR_DIR/$ORT_PKG.tgz"
fi

# Locate the shared library (extension differs per platform).
ORT_LIB="$(find "$ORT_DIR/lib" -maxdepth 1 -name 'libonnxruntime.so' -o -name 'libonnxruntime.dylib' 2>/dev/null | head -n1)"

# --- Model ---------------------------------------------------------------------
if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading hand-landmark model ..."
  mkdir -p "$MODEL_DIR"
  curl -fSL --retry 4 --retry-delay 2 -o "$MODEL_PATH" "$MODEL_URL"
fi

echo
echo "Setup complete."
echo "  Model:   $MODEL_PATH"
echo "  Runtime: $ORT_LIB"
echo
echo "Run the spike with:"
echo "  export ORT_DYLIB_PATH=\"$ORT_LIB\""
echo "  cargo run"

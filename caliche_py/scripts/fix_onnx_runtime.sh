#!/bin/bash
# Post-build script to fix ONNX Runtime library paths for Python extension
# This script copies the ONNX Runtime library and sets rpaths on macOS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

# Find the Python extension location
# For development installs (maturin develop), the extension is in the python package
PYTHON_PKG_DIR="$PROJECT_DIR/python/caliche"

# Find the built extension
if [ -f "$PYTHON_PKG_DIR/_caliche_py.abi3.so" ]; then
    EXT_PATH="$PYTHON_PKG_DIR/_caliche_py.abi3.so"
elif [ -f "$PYTHON_PKG_DIR/_caliche_py.so" ]; then
    EXT_PATH="$PYTHON_PKG_DIR/_caliche_py.so"
else
    echo "Warning: Could not find Python extension, skipping ONNX Runtime fix"
    exit 0
fi

# Find ONNX Runtime library
ONNX_LIB=""
if [ -f "$ROOT_DIR/target/release/libonnxruntime.1.16.0.dylib" ]; then
    ONNX_LIB="$ROOT_DIR/target/release/libonnxruntime.1.16.0.dylib"
elif [ -f "$ROOT_DIR/target/release/libonnxruntime.dylib" ]; then
    ONNX_LIB="$ROOT_DIR/target/release/libonnxruntime.dylib"
else
    # Try to find in build directory
    ONNX_LIB=$(find "$ROOT_DIR/target/release/build" -name "libonnxruntime.1.16.0.dylib" -not -path "*/dSYM/*" -type f | head -1)
fi

if [ -z "$ONNX_LIB" ] || [ ! -f "$ONNX_LIB" ]; then
    echo "Warning: Could not find ONNX Runtime library, skipping fix"
    echo "This is OK if you're not using local-ort features"
    exit 0
fi

# Copy library to Python package directory
echo "Copying ONNX Runtime library to Python package..."
cp "$ONNX_LIB" "$PYTHON_PKG_DIR/libonnxruntime.1.16.0.dylib" 2>/dev/null || true

# On macOS, fix rpath
if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v install_name_tool &> /dev/null; then
        echo "Setting rpath for Python extension..."
        install_name_tool -add_rpath @loader_path "$EXT_PATH" 2>/dev/null || true
        
        # Also update the library's install name if needed
        if [ -f "$PYTHON_PKG_DIR/libonnxruntime.1.16.0.dylib" ]; then
            install_name_tool -id @rpath/libonnxruntime.1.16.0.dylib "$PYTHON_PKG_DIR/libonnxruntime.1.16.0.dylib" 2>/dev/null || true
        fi
    fi
fi

echo "ONNX Runtime fix completed successfully"


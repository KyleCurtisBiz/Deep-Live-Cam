# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep-Live-Cam is a real-time face swapping application that uses AI models to swap faces in images, videos, and live webcam streams. The application is built with Python and uses PyTorch, InsightFace, and ONNX Runtime for AI inference.

## Commands

### Running the Application

```bash
# Basic execution (CPU)
python run.py

# With CUDA (NVIDIA GPU)
python run.py --execution-provider cuda

# On macOS with Apple Silicon
python3.10 run.py --execution-provider coreml

# Common usage with source and target
python run.py -s path/to/source.jpg -t path/to/target.mp4 -o output.mp4
```

### Development Setup

```bash
# Create virtual environment (Python 3.10 required for macOS)
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models to models/ directory:
# - inswapper_128_fp16.onnx
# - GFPGANv1.4.pth
```

### Type Checking

```bash
# Run MyPy type checking
mypy . --config-file mypi.ini
```

## Architecture

### Core Components

1. **Entry Point**: `run.py` → `modules.core.run()`
   - Handles CLI argument parsing and initializes the application

2. **Processing Pipeline**:
   - `modules/face_analyser.py`: Detects and analyzes faces using InsightFace
   - `modules/processors/frame/face_swapper.py`: Core face swapping logic using inswapper model
   - `modules/processors/frame/face_enhancer.py`: Optional face enhancement using GFPGAN

3. **GUI System**: `modules/ui.py`
   - Built with CustomTkinter
   - Persistent state management for user preferences
   - Real-time preview capabilities

4. **Global State**: `modules/globals.py`
   - Manages execution providers (CPU, CUDA, CoreML, etc.)
   - Stores processing options and configurations

### Processing Flow

1. Source face embedding extraction
2. Target face detection in each frame
3. Face swapping using ONNX model
4. Optional enhancement and post-processing
5. Frame reassembly with audio preservation

### Key Design Patterns

- **Single-threaded ONNX execution** for optimal CUDA performance
- **Modular processor system** allowing chaining of face_swapper and face_enhancer
- **Lazy model loading** to reduce memory usage
- **Temp file management** with automatic cleanup

## Development Guidelines

### Branching Strategy

- `main`: Production branch
- `premain`: Testing branch - all changes must go here first
- `experimental`: For large or disruptive changes

### Testing Requirements

When making changes, manually test:
1. Real-time face swap with enhancer on/off
2. Face mapping functionality
3. Camera listing and switching
4. Real-time FPS (should maintain 30+ FPS)
5. GPU stress test (15+ minutes)
6. Application responsiveness during processing

### Platform Considerations

- **macOS**: Must use Python 3.10 (not 3.11+) and install `python-tk@3.10`
- **Windows**: Requires Visual Studio 2022 Runtimes
- **CUDA**: Requires CUDA Toolkit 12.8.0 and cuDNN v8.9.7
- **Memory**: Platform-specific memory limits configured in `modules/core.py`

### Performance Optimization

- Use `--execution-threads 1` for better GPU performance
- Monitor GPU memory usage with temp file limits
- Face detection caching to avoid redundant computations
- Proper resource cleanup after processing

## Performance Investigation History (macOS Apple Silicon)

### Issue Summary
Initial problem: Real-time face swapping achieving only ~1 FPS on Apple Silicon Mac despite moderate CPU (30%) and GPU (50%) usage.

### Investigation Timeline & Findings

#### 1. Initial Diagnosis Attempts
**Attempted Fix**: Updated ONNX Runtime from 1.13.1 to 1.16.3
- **Result**: No performance improvement
- **Learning**: Version wasn't the bottleneck

#### 2. Threading Limitation Discovery
**Root Cause Found**: `OMP_NUM_THREADS=1` was being applied to ALL execution providers, not just CUDA
- **Fix Applied**: Modified `modules/core.py` line 4-5 to only apply single-threading to CUDA
- **Code Change**: `if any(arg.startswith('--execution-provider') and 'cuda' in arg for arg in sys.argv)`
- **Result**: Allowed CoreML to use multiple threads

#### 3. Face Detection Size Optimization  
**Attempted Fix**: Reduced detection size from 640x640 to 320x320, then to 256x256
- **File**: `modules/face_analyser.py` line 23
- **Reasoning**: Smaller detection = less computation
- **Result**: Minimal impact on CPU performance, but enabled CoreML compatibility

#### 4. CoreML Compatibility Investigation
**Critical Discovery**: CoreML was failing with shape mismatch error:
```
CoreML static output shape ({1,1,1,800,1}) and inferred shape ({3200,1}) have different ranks
```
- **Root Cause**: Detection size of 640x640 incompatible with CoreML provider
- **Fix**: Reduced to det_size=(256, 256) which CoreML accepts
- **Result**: CoreML execution now works without crashes

#### 5. Performance Profiling with Timing Logs
**Added comprehensive timing to identify actual bottlenecks**:
- **Files Modified**: `modules/ui.py`, `modules/processors/frame/face_swapper.py`
- **Timing Points Added**:
  - Camera capture time
  - Frame preparation time  
  - Face detection time
  - Face swapping time
  - UI rendering time
  - Total loop time

**Key Performance Metrics Discovered**:
- **Without face present**: 30 FPS (proves pipeline can be fast)
- **With face present**: 1 FPS (bottleneck identified)
- **Face detection**: ~0.09s (acceptable)
- **Face swapping**: ~1.0s (PRIMARY BOTTLENECK)
- **UI rendering**: ~0.03s (minimal)

#### 6. ONNX Runtime Optimization Attempts
**Applied provider-specific optimizations**:
- **CPU Provider**: Added memory arena, thread configuration
- **CoreML Provider**: Added `use_cpu_only: false`, `require_static_input_shapes: false`
- **Graph Optimization**: Enabled `ORT_ENABLE_ALL`
- **Result**: Minimal improvement (~50ms reduction)

### Current State & Conclusions

#### What Works
1. **CoreML execution provider** now functions without crashes (det_size=256x256)
2. **Multi-threading** enabled for non-CUDA providers
3. **Performance monitoring** provides detailed bottleneck identification
4. **30 FPS achieved** when no face is present (proves pipeline capability)

#### Primary Bottleneck Identified
**Face swapping inference takes ~1.0 second per frame** - this is the core issue
- **inswapper_128_fp16.onnx model** running on CPU is extremely slow
- **CoreML provider** may still not be optimally accelerating the face swap model
- **Hardware acceleration** not effectively utilized for the inference step

#### Remaining Investigation Areas
1. **Model-specific acceleration**: inswapper model may need different optimization
2. **CoreML vs CPU comparison**: Need timing comparison between providers
3. **Model quantization**: Could use lighter/faster face swap model
4. **Frame skipping**: Could process every 2nd-3rd frame for real-time performance
5. **Asynchronous processing**: Could implement producer-consumer pattern

#### Commands for Testing
```bash
# Test CoreML performance (should work now)
python3.10 run.py --execution-provider coreml --execution-threads 4

# Test CPU performance with optimizations
python3.10 run.py --execution-provider cpu --execution-threads 8

# Monitor timing output in console for bottleneck analysis
```

#### Technical Notes
- **ONNX Runtime Version**: 1.16.3 (latest)
- **Python Version**: 3.10 (required for macOS compatibility)
- **Face Detection Size**: 256x256 (CoreML compatible)
- **Primary Model**: inswapper_128_fp16.onnx (face swapping)
- **Secondary Model**: GFPGANv1.4.pth (enhancement, uses MPS acceleration)
Building OpenCV with NVIDIA nvcuvid (Video Decode) Support {#tutorial_nvcuvid_build}
==========================================================

|    |    |
| -: | :- |
| Compatibility | OpenCV >= 4.0 |

@tableofcontents

# Overview {#tutorial_nvcuvid_build_overview}

nvcuvid is the NVIDIA Video Decode API, part of the NVIDIA Video Codec SDK. It exposes hardware-accelerated video decoding on NVIDIA GPUs via the CUVID interface. When OpenCV is built with nvcuvid support, the `videoio` module can decode video streams directly on the GPU, bypassing the CPU for decode work and reducing latency.

Use cases where hardware video decode matters:
- High-throughput video analytics pipelines (many simultaneous streams)
- Real-time inference applications where GPU decode feeds directly into GPU-side pre-processing
- Transcoding and streaming applications that benefit from keeping data on the GPU

The nvcuvid path in OpenCV is gated behind two CMake options:

| Option | Default when CUDA is ON | Purpose |
| :----- | :---------------------- | :------ |
| `WITH_NVCUVID` | `ON` | Hardware video *decode* (this guide) |
| `WITH_NVCUVENC` | `ON` | Hardware video *encode* (nvEncodeAPI) |

This guide covers `WITH_NVCUVID` only.

# Prerequisites {#tutorial_nvcuvid_build_prereqs}

## Hardware

Any NVIDIA GPU that supports CUVID hardware decoding. Fermi (compute 2.0) and later GPUs include a dedicated video decode engine. Check [NVIDIA's codec support matrix](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new) for codec-specific support by GPU generation.

## Software

### 1. NVIDIA display driver

Install the latest NVIDIA display driver for your operating system from [https://www.nvidia.com/drivers](https://www.nvidia.com/drivers). The shared library `libnvcuvid.so` (Linux) or `nvcuvid.dll` (Windows) ships with the driver, not the CUDA Toolkit.

Minimum recommended driver version: **418.30** (Linux) / **418.81** (Windows), which introduced the Video Codec SDK 9.0 API revision. Using the latest available driver is strongly recommended.

### 2. CUDA Toolkit

Download and install the CUDA Toolkit (version 10.0 or later recommended) from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

The CUDA Toolkit provides `nvcc`, the CUDA runtime headers, and (on some platforms) a stub `nvcuvid.so` under `<cuda_root>/lib/stubs/`.

### 3. NVIDIA Video Codec SDK headers

The header `nvcuvid.h` is **not** included in the CUDA Toolkit. It must be obtained separately from the NVIDIA Video Codec SDK:

1. Download the Video Codec SDK from [https://developer.nvidia.com/nvidia-video-codec-sdk](https://developer.nvidia.com/nvidia-video-codec-sdk) (free download, requires NVIDIA developer account).
2. Extract the archive. The relevant files are:
   - `Interface/nvcuvid.h` — the decode API header
   - `Interface/cuviddec.h` — supporting decode structures header
   - `Lib/linux/stubs/x86_64/libnvcuvid.so` — link-time stub (Linux, for build environments without a driver)
   - `Lib/Win32/` or `Lib/x64/` — import libraries (Windows)

3. Copy the headers into the CUDA Toolkit include directory so CMake can find them automatically:

    ```sh
    sudo cp /path/to/Video_Codec_SDK_X.X.X/Interface/nvcuvid.h  /usr/local/cuda/include/
    sudo cp /path/to/Video_Codec_SDK_X.X.X/Interface/cuviddec.h /usr/local/cuda/include/
    ```

    Alternatively, you can place them anywhere and point CMake at the location (see the CMake configuration section below).

### 4. Build tools

CMake 3.18 or later and a C++17-capable compiler are required. See @ref tutorial_linux_install or @ref tutorial_windows_install for platform-specific setup.

# CMake Configuration {#tutorial_nvcuvid_build_cmake}

## Minimal configuration (Linux)

```sh
cmake \
  -DWITH_CUDA=ON \
  -DWITH_NVCUVID=ON \
  -DCMAKE_BUILD_TYPE=Release \
  /path/to/opencv
```

`WITH_NVCUVID` is `ON` by default whenever `WITH_CUDA=ON`, so passing it explicitly is only needed if you previously disabled it.

## Specifying non-standard header or library paths

If `nvcuvid.h` is not under the CUDA Toolkit `include/` directory, tell CMake where to search via `CUDA_TOOLKIT_ROOT_DIR` or by copying the headers as described above.

If the `nvcuvid` library is not found automatically (common in Docker or CI environments that have no GPU driver installed), point CMake at the stub library from the Video Codec SDK:

```sh
cmake \
  -DWITH_CUDA=ON \
  -DWITH_NVCUVID=ON \
  -DCUDA_nvcuvid_LIBRARY=/path/to/Video_Codec_SDK_X.X.X/Lib/linux/stubs/x86_64/libnvcuvid.so \
  -DCMAKE_BUILD_TYPE=Release \
  /path/to/opencv
```

On Windows the equivalent is:

```sh
cmake \
  -DWITH_CUDA=ON \
  -DWITH_NVCUVID=ON \
  -DCUDA_nvcuvid_LIBRARY=C:\Video_Codec_SDK_X.X.X\Lib\x64\nvcuvid.lib \
  -DCMAKE_BUILD_TYPE=Release \
  /path/to/opencv
```

## Full example (Linux, Release build with contrib)

```sh
mkdir build && cd build

cmake \
  -DWITH_CUDA=ON \
  -DWITH_NVCUVID=ON \
  -DWITH_NVCUVENC=ON \
  -DCUDA_ARCH_BIN="8.6" \
  -DOPENCV_EXTRA_MODULES_PATH=/path/to/opencv_contrib/modules \
  -DBUILD_opencv_cudacodec=ON \
  -DCMAKE_BUILD_TYPE=Release \
  /path/to/opencv

cmake --build . --parallel $(nproc)
```

Replace `8.6` with the compute capability of your GPU (e.g. `7.5` for Turing, `8.0` for Ampere A100, `9.0` for Hopper). You can look up your GPU's compute capability at [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus).

@note The `cudacodec` module lives in `opencv_contrib`. To use hardware decode through the `cv::cudacodec::VideoReader` API you must also build with `-DOPENCV_EXTRA_MODULES_PATH` pointing to the contrib modules and `-DBUILD_opencv_cudacodec=ON`.

# Verifying the Build {#tutorial_nvcuvid_build_verify}

## Check the CMake summary

After running CMake, look for the following lines in the configuration summary printed to the terminal:

```
--   NVIDIA CUDA
--     Use CUFFT:                   YES
--     Use CUBLAS:                  YES
--     NVCUVID:                     YES (/usr/lib/x86_64-linux-gnu/libnvcuvid.so)
--     NVCUVENC:                    YES (...)
```

If `NVCUVID` shows `NO` or is absent:
- **Header not found**: copy `nvcuvid.h` and `cuviddec.h` into `<cuda_root>/include/` as described above.
- **Library not found**: set `CUDA_nvcuvid_LIBRARY` to the stub path or the driver-provided library.

## Check the built library

On Linux, confirm the `cudacodec` module was built and linked against nvcuvid:

```sh
# List the installed module
ls install/lib/libopencv_cudacodec*.so

# Check linkage
ldd install/lib/libopencv_cudacodec*.so | grep nvcuvid
```

Expected output (the exact path varies by system):

```
    libnvcuvid.so.1 => /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1
```

## Programmatic check (Python)

```python
import cv2
print(cv2.getBuildInformation())
```

Look for `NVCUVID: YES` in the CUDA section of the output.

## Functional test

```python
import cv2

cap = cv2.cudacodec.createVideoReader("test_video.mp4")
ret, frame = cap.nextFrame()
if ret:
    print("Hardware decode succeeded, frame shape:", frame.size())
else:
    print("Hardware decode failed")
```

If `cv2.cudacodec` is not available, either the contrib modules were not built or nvcuvid was not found at configure time.

# Common Issues {#tutorial_nvcuvid_build_troubleshooting}

| Symptom | Likely cause | Fix |
| :------ | :----------- | :-- |
| `NVCUVID: Header not found` during CMake | `nvcuvid.h` not in CUDA include path | Copy headers from Video Codec SDK into `<cuda_root>/include/` |
| `NVCUVID: Library not found` during CMake | Driver not installed or build machine has no GPU | Set `CUDA_nvcuvid_LIBRARY` to the stub `.so`/`.lib` from the SDK |
| `ImportError: cv2.cudacodec not found` at runtime | contrib not built or nvcuvid disabled | Rebuild with `-DOPENCV_EXTRA_MODULES_PATH=...` and `-DWITH_NVCUVID=ON` |
| `cv2.error: NVCUVID is not supported` at runtime | nvcuvid stub used at build time but driver absent at runtime | Install NVIDIA driver on the target machine |
| Decode works but frames appear corrupt | GPU architecture mismatch | Rebuild with the correct `CUDA_ARCH_BIN` for your GPU |

# See Also {#tutorial_nvcuvid_build_see_also}

- @ref tutorial_general_install — general OpenCV build options
- @ref tutorial_config_reference — full CMake option reference
- @ref tutorial_linux_install — Linux installation guide
- @ref tutorial_building_tegra_cuda — building for NVIDIA Tegra/Jetson platforms
- [NVIDIA Video Codec SDK documentation](https://docs.nvidia.com/video-technologies/video-codec-sdk/index.html)
- [NVIDIA codec support matrix](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new)

# Bulls-And-Cows
Using CUDA and two GPUs to run Bulls and Cows.

---

## Project Overview

This project implements the classic Bulls and Cows game optimized with CUDA to leverage GPU acceleration. It supports running on two GPUs for enhanced performance.

---

## Requirements

- CMake (version 3.18 or higher)  
- NVIDIA CUDA Toolkit installed  
- Compatible NVIDIA GPUs (at least 2 for dual-GPU usage)  
- C++17 compatible compiler  

---

## Build Instructions

This project uses CMake to configure and build the code. You can use the provided `Makefile` to simplify building and running.

### Basic build and run

```bash
$ make
```
This will build the project inside a `build/` directory and run the executable automatically.

Other useful commands
- Build only (no run):
    ```bash
    $ make build
    ```
- Run only
    ```bash
    $ make run
    ```
- Clean build directory:
    ```bash
    $ make clean
    ```

### Project Structure
```sh
.
├── CMakeLists.txt
├── Makefile
├── include/
├── src/
│   ├── main.cpp
│   ├── player.cpp
│   ├── utils.cpp
│   └── cuda_utils.cu
└── build/   # Build output directory (auto-generated)

```

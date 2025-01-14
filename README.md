
# Test CUDA Compilation

## On Windows
- Install the CUDA toolkit
- Clone this repository
- Open as an administrator the x64 Native Tools Command
- Reach the cloned folder
- Build and run:
```console
mkdir build_debug
cd build_debug
cmake -DCMAKE_BUILD_TYPE=Debug -G "Visual Studio 15 2017" -A x64 ..
cmake --build .
.\bin\Debug\Debug\TestCompilation.exe
```

Note:
- To use a newer version of CMake that the one provide with VS: `set PATH=C:\Program Files\CMake\bin;%PATH%`
- `-A x64` is used to specify the 64 bits arch that is the only one compatible with CUDA
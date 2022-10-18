## Creating CustomOps in ORT

Define the library similar to `custom_op_library.h` and `custom_op_library.cc`. This uses the ORT C (and C++) APIs to create a new domain, create new operators with their corresponding kernels, and register them in the newly created domain. This should expose `RegisterCustomOps` method as done in `custom_op_library.h`.

Then, compile the source into a shared library (`.so` or `.dll` depending on the target platform). Use CMake for this and have a look at the `CMakeLists.txt` file.

Once the source is compiled to a shared library, this can be loaded into an ORT session and can be used. Look at `load_custom_op.py` to see how to do that.

## Steps to Run
```bash
cd build
cmake .. ## generates the Makefile
cmake --build . ## compiles and generates the .sh file
```
# Convolution3D
Convolution3D is an Avisynth filter that will apply a 3D - temporal and/or spatial - convolution to a frame

Build instructions
==================
VS2019: 
  use IDE

Windows GCC (mingw installed by msys2):
  from the 'build' folder under project root:

  del ..\CMakeCache.txt
  cmake .. -G "MinGW Makefiles" -DENABLE_INTEL_SIMD:bool=on
  @rem test: cmake .. -G "MinGW Makefiles" -DENABLE_INTEL_SIMD:bool=off
  cmake --build . --config Release  

Linux
  from the 'build' folder under project root:
  ENABLE_INTEL_SIMD is automatically off for non x86 arhitectures
  
* Clone repo and build
    
        git clone https://github.com/pinterf/Convolution3D
        cd Convolution3D
        cmake -B build -S .
        cmake --build build

  Useful hints:        
   build after clean:

        cmake --build build --clean-first

   Force no asm support

        cmake -B build -S . -DENABLE_INTEL_SIMD:bool=off

   delete cmake cache

        rm build/CMakeCache.txt

* Find binaries at
    
        build/Convolution3D/libconvolution3d.so

* Install binaries

        cd build
        sudo make install
  


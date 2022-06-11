#pragma once

#define INIT_X_POS 128   // Window position x
#define INIT_Y_POS 128   // window position y
#define INIT_WIDTH 1024  // window width
#define INIT_HEIGHT 1024 // window height

#define DIM 512
#define NUM_PRTICLES (DIM * DIM)  // Total prticles size
#define CPADW (DIM / 2 + 1)       // Padded width for real->complex in-place FFT
#define RPADW (2 * (DIM / 2 + 1)) // Padded width for real->complex in-place FFT
#define PDS (DIM * CPADW)         // Padded total prticles size

#define DT 0.09f           // Delta T for interative solver
#define VISCOSITY 0.0025f  // Viscosity constant
#define FORCE (5.8f * DIM) // Force scale factor
#define FR 4               // Force update radius

#define TILEX 64 // Tile width
#define TILEY 64 // Tile height
#define TIDSX 64 // Tids in X
#define TIDSY 4  // Tids in Y

// OpenGL Graphics includes
#include <GL/freeglut.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA FFT Libraries
#include <cufft.h>

// CUDA helper functions
#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <helper_cuda.h>

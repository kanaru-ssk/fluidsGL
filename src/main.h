/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
#ifndef DEFINES_H
#define DEFINES_H

#define INIT_X_POS 128
#define INIT_Y_POS 128
#define INIT_WIDTH 1024
#define INIT_HEIGHT 1024

#define DIM 512       // Square size of solver domain
#define DS    (DIM*DIM)  // Total domain size
#define CPADW (DIM/2+1)  // Padded width for real->complex in-place FFT
#define RPADW (2*(DIM/2+1))  // Padded width for real->complex in-place FFT
#define PDS   (DIM*CPADW) // Padded total domain size

#define DT     0.09f     // Delta T for interative solver
#define VIS    0.0025f   // Viscosity constant
#define FORCE (5.8f*DIM) // Force scale factor 
#define FR     4         // Force update radius

#define TILEX 64 // Tile width
#define TILEY 64 // Tile height
#define TIDSX 64 // Tids in X
#define TIDSY 4  // Tids in Y

#endif

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

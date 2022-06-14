#pragma once

#define INIT_X_POS 128   // Window position x
#define INIT_Y_POS 1024  // window position y
#define INIT_WIDTH 1024  // window width
#define INIT_HEIGHT 1024 // window height

#define FIELD_DIV 512                        // 速度場の分割数
#define CELL_SCALE (1 / FIELD_DIV)           // セルの幅
#define NUM_PRTICLES (FIELD_DIV * FIELD_DIV) // 粒子数

#define CPADW (FIELD_DIV / 2 + 1) // Padded width for real->complex in-place FFT
#define PDS (FIELD_DIV * CPADW)   // Padded total prticles size

#define DT 0.09f         // Delta T for interative solver
#define VISCOSITY 0.025f // Viscosity constant
#define FORCE 256        // Force scale factor
#define FORCE_RADIUS 16  // Force update radius

#define GRID_W 64  // grid size of x
#define GRID_H 64  // grid size of y
#define BLOCK_W 64 // block size of x
#define BLOCK_H 8  // block size of y

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

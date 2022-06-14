#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cufft.h>		 // CUDA FFT Libraries
#include <helper_cuda.h> // Helper functions for CUDA Error handling

// OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

// FluidsGL CUDA kernel definitions
#include "fluidsGL_kernels.cuh"

// Texture object for reading velocity field
cudaTextureObject_t texObj;
static cudaArray *array = NULL;

// Particle data
extern GLuint vbo;									   // OpenGL vertex buffer object
extern struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange

// Texture pitch
extern size_t tPitch;
extern cufftHandle planr2c;
extern cufftHandle planc2r;
float2 *vxfield = NULL;
float2 *vyfield = NULL;

void setupTexture(int x, int y)
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

	cudaMallocArray(&array, &desc, y, x);
	getLastCudaError("cudaMalloc failed");

	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = array;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeWrap;
	texDescr.readMode = cudaReadModeElementType;

	checkCudaErrors(cudaCreateTextureObject(&texObj, &texRes, &texDescr, NULL));
}

void updateTexture(float2 *data, size_t wib, size_t h, size_t pitch)
{
	checkCudaErrors(cudaMemcpy2DToArray(array, 0, 0, data, pitch, wib, h, cudaMemcpyDeviceToDevice));
}

void deleteTexture(void)
{
	checkCudaErrors(cudaDestroyTextureObject(texObj));
	checkCudaErrors(cudaFreeArray(array));
}

// Note that these kernels are designed to work with arbitrary
// domain sizes, not just domains that are multiples of the tile
// size. Therefore, we have extra code that checks to make sure
// a given thread location falls within the domain boundaries in
// both X and Y. Also, the domain is covered by looping over
// multiple elements in the Y direction, while there is a one-to-one
// mapping between threads in X and the tile size in X.
// Nolan Goodnight 9/22/06

// This method adds constant force vectors to the velocity field
// stored in 'v' according to v(x,t+1) = v(x,t) + dt * f.
// 外力が加わった時の処理
extern "C" void addForces(float2 *v, int posX, int posY, float forceX, float forceY)
{

	dim3 block(FORCE_RADIUS, FORCE_RADIUS);

	d_addForces<<<1, block>>>(v, posX, posY, forceX, forceY, tPitch);
	getLastCudaError("addForces_k failed.");
}

__global__ void d_addForces(float2 *v, int posX, int posY, float forceX, float forceY, size_t pitch)
{
	int forcePosX = threadIdx.x - blockDim.x / 2;
	int forcePosY = threadIdx.y - blockDim.y / 2;
	// 速度配列内のグローバルスレッド位置
	float2 *vfield = (float2 *)((char *)v + (posY + forcePosY) * pitch) + posX + forcePosX;

	float2 vterm = *vfield;
	vterm.x += forceX;
	vterm.y += forceY;

	*vfield = vterm;
}

// This method performs the velocity advection step, where we
// trace velocity vectors back in time to update each grid cell.
// That is, v(x,t+1) = v(p(x,-dt),t). Here we perform bilinear
// interpolation in the velocity space.

// 移流を計算
extern "C" void advectVelocity(float2 *v)
{
	dim3 grid((FIELD_DIV / BLOCK_W) + (!(FIELD_DIV % BLOCK_W) ? 0 : 1), (FIELD_DIV / BLOCK_H) + (!(FIELD_DIV % BLOCK_H) ? 0 : 1));

	dim3 block(BLOCK_W, BLOCK_H);

	updateTexture(v, FIELD_DIV * sizeof(float2), FIELD_DIV, tPitch);
	d_advectVelocity<<<grid, block>>>(v, GRID_H / BLOCK_H, texObj, tPitch);

	getLastCudaError("d_advectVelocity failed.");
}

__global__ void d_advectVelocity(float2 *v, int lb, cudaTextureObject_t texObject, size_t pitch)
{

	int cellX = blockDim.x * blockIdx.x + threadIdx.x;
	int cellY = blockDim.y * blockIdx.y + threadIdx.y;

	float2 vterm, ploc;

	if (cellX < FIELD_DIV && cellY < FIELD_DIV)
	{
		vterm = tex2D<float2>(texObject, (float)cellX, (float)cellY);
		ploc.x = (cellX + 0.5f) - (DT * vterm.x * FIELD_DIV);
		ploc.y = (cellY + 0.5f) - (DT * vterm.y * FIELD_DIV);
		vterm = tex2D<float2>(texObject, ploc.x, ploc.y);

		float2 *vfield = (float2 *)((char *)v + cellY * pitch) + cellX;
		*vfield = vterm;
	}
}

// This method performs velocity diffusion and forces mass conservation
// in the frequency domain. The inputs 'vx' and 'vy' are complex-valued
// arrays holding the Fourier coefficients of the velocity field in
// X and Y. Diffusion in this space takes a simple form described as:
// v(k,t) = v(k,t) / (1 + visc * dt * k^2), where visc is the viscosity,
// and k is the wavenumber. The projection step forces the Fourier
// velocity vectors to be orthogonal to the vectors for each
// wavenumber: v(k,t) = v(k,t) - ((k dot v(k,t) * k) / k^2.

// 拡散を計算
extern "C" void diffuseProject(float2 *vx, float2 *vy, int dx, int dy, float dt, float visc)
{
	// Forward FFT
	checkCudaErrors(cufftExecR2C(planr2c, (cufftReal *)vx, (cufftComplex *)vx));
	checkCudaErrors(cufftExecR2C(planr2c, (cufftReal *)vy, (cufftComplex *)vy));

	uint3 grid = make_uint3((dx / GRID_W) + (!(dx % GRID_W) ? 0 : 1),
							(dy / GRID_H) + (!(dy % GRID_H) ? 0 : 1), 1);
	uint3 block = make_uint3(BLOCK_W, BLOCK_H, 1);

	d_diffuseProject<<<grid, block>>>(vx, vy, dx, dy, dt, visc, GRID_H / BLOCK_H);
	getLastCudaError("diffuseProject_k failed.");

	// Inverse FFT
	checkCudaErrors(cufftExecC2R(planc2r, (cufftComplex *)vx, (cufftReal *)vx));
	checkCudaErrors(cufftExecC2R(planc2r, (cufftComplex *)vy, (cufftReal *)vy));
}

__global__ void d_diffuseProject(float2 *vx, float2 *vy, int dx, int dy, float dt, float visc, int lb)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
	int p;

	float2 xterm, yterm;

	// gtidx is the domain location in x for this thread
	if (gtidx < dx)
	{
		for (p = 0; p < lb; p++)
		{
			// fi is the domain location in y for this thread
			int fi = gtidy + p;

			if (fi < dy)
			{
				int fj = fi * dx + gtidx;
				xterm = vx[fj];
				yterm = vy[fj];

				// Compute the index of the wavenumber based on the
				// data order produced by a standard NN FFT.
				int iix = gtidx;
				int iiy = (fi > dy / 2) ? (fi - (dy)) : fi;

				// Velocity diffusion
				float kk = (float)(iix * iix + iiy * iiy); // k^2
				float diff = 1.f / (1.f + visc * dt * kk);
				xterm.x *= diff;
				xterm.y *= diff;
				yterm.x *= diff;
				yterm.y *= diff;

				// Velocity projection
				if (kk > 0.f)
				{
					float rkk = 1.f / kk;
					// (実数部分) Real portion of velocity projection
					float rkp = (iix * xterm.x + iiy * yterm.x);
					// (虚数部分) Imaginary portion of velocity projection
					float ikp = (iix * xterm.y + iiy * yterm.y);
					xterm.x -= rkk * rkp * iix;
					xterm.y -= rkk * ikp * iix;
					yterm.x -= rkk * rkp * iiy;
					yterm.y -= rkk * ikp * iiy;
				}

				vx[fj] = xterm;
				vy[fj] = yterm;
			}
		}
	}
}

// This method updates the velocity field 'v' using the two complex
// arrays from the previous step: 'vx' and 'vy'. Here we scale the
// real components by 1/(dx*dy) to account for an unnormalized FFT.

extern "C" void updateVelocity(float2 *v, float *vx, float *vy, int dx, int pdx, int dy)
{
	dim3 grid((dx / GRID_W) + (!(dx % GRID_W) ? 0 : 1), (dy / GRID_H) + (!(dy % GRID_H) ? 0 : 1));
	dim3 block(BLOCK_W, BLOCK_H);

	d_updateVelocity<<<grid, block>>>(v, vx, vy, dx, pdx, dy, GRID_H / BLOCK_H, tPitch);
	getLastCudaError("updateVelocity_k failed.");
}

__global__ void d_updateVelocity(float2 *v, float *vx, float *vy, int dx, int pdx, int dy, int lb, size_t pitch)
{

	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
	int p;

	float vxterm, vyterm;
	float2 nvterm;

	// gtidx is the domain location in x for this thread
	if (gtidx < dx)
	{
		for (p = 0; p < lb; p++)
		{
			// fi is the domain location in y for this thread
			int fi = gtidy + p;

			if (fi < dy)
			{
				int fjr = fi * pdx + gtidx;
				vxterm = vx[fjr];
				vyterm = vy[fjr];

				// Normalize the result of the inverse FFT
				float scale = 1.f / (dx * dy);
				nvterm.x = vxterm * scale;
				nvterm.y = vyterm * scale;

				float2 *fj = (float2 *)((char *)v + fi * pitch) + gtidx;
				*fj = nvterm;
			}
		} // If this thread is inside the domain in Y
	}	  // If this thread is inside the domain in X
}

// This method updates the particles by moving particle positions
// according to the velocity field and time step. That is, for each
// particle: p(t+1) = p(t) + dt * v(p(t)).

extern "C" void advectParticles(GLuint vbo, float2 *v, int dx, int dy, float dt)
{
	dim3 grid((dx / GRID_W) + (!(dx % GRID_W) ? 0 : 1), (dy / GRID_H) + (!(dy % GRID_H) ? 0 : 1));
	dim3 block(BLOCK_W, BLOCK_H);

	float2 *p;
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsMapResources failed");

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&p, &num_bytes,
										 cuda_vbo_resource);
	getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

	d_advectParticles<<<grid, block>>>(p, v, dx, dy, dt, GRID_H / BLOCK_H, tPitch);
	getLastCudaError("advectParticles_k failed.");

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	getLastCudaError("cudaGraphicsUnmapResources failed");
}

__global__ void d_advectParticles(float2 *part, float2 *v, int dx, int dy, float dt, int lb, size_t pitch)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
	int p;

	// gtidx is the domain location in x for this thread
	float2 pterm, vterm;

	if (gtidx < dx)
	{
		for (p = 0; p < lb; p++)
		{
			// fi is the domain location in y for this thread
			int fi = gtidy + p;

			if (fi < dy)
			{
				int fj = fi * dx + gtidx;
				pterm = part[fj];

				// 粒子が速度場のどこに存在するか算出
				int xvi = ((int)(pterm.x * dx));
				int yvi = ((int)(pterm.y * dy));

				// 粒子の位置の速度を抽出
				vterm = *((float2 *)((char *)v + yvi * pitch) + xvi);

				// pterm.x += dt * vterm.x;
				// pterm.x *= (-2 * (0 < pterm.x) + 1); // 0 < pterm.x の時反発
				// vterm.x *= (-2 * (0 < pterm.x) + 1);
				// pterm.x -= ((pterm.x < 1) * 2 * (pterm.x - (int)pterm.x)); // pterm.x < 1 の時反発
				// vterm.x *= (-2 * (pterm.x < 1) + 1);

				// pterm.y += dt * vterm.y;
				// pterm.y *= (-2 * (0 < pterm.y) + 1); // 0 < pterm.x の時反発
				// vterm.y *= (-2 * (0 < pterm.y) + 1);
				// pterm.y -= ((pterm.y < 1) * 2 * (pterm.y - (int)pterm.y)); // pterm.x < 1 の時反発
				// vterm.y *= (-2 * (pterm.y < 1) + 1);

				pterm.x += dt * vterm.x;		  // 経過時間 * x方向速度
				pterm.x = pterm.x - (int)pterm.x; // 整数部分削除 (-1.0 ~ 1.0)
				pterm.x += 1.f;					  // (0.0 ~ 2.0)
				pterm.x = pterm.x - (int)pterm.x; // 整数部分削除 (0.0 ~ 1.0)
				pterm.y += dt * vterm.y;
				pterm.y = pterm.y - (int)pterm.y;
				pterm.y += 1.f;
				pterm.y = pterm.y - (int)pterm.y;

				part[fj] = pterm;
			}
		} // If this thread is inside the domain in Y
	}	  // If this thread is inside the domain in X
}

__global__ void addForces_k(float2 *v, int dx, int dy, int posX, int posY, float forceX, float forceY, int r, size_t pitch)
{

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float2 *fj = (float2 *)((char *)v + (ty + posY) * pitch) + tx + posX; // 速度配列内のグローバルスレッド位置

	float2 vterm = *fj;
	tx -= r;
	ty -= r;

	// 滑らかにするために減衰を計算
	float s = 1.f / (1.f + tx * tx * tx * tx + ty * ty * ty * ty);

	vterm.x += s * forceX;
	vterm.y += s * forceY;
	// vterm.x += forceX;
	// vterm.y += forceY;
	*fj = vterm;
}

__global__ void advectVelocity_k(float2 *v, float *vx, float *vy, int dx, int pdx, int dy, float dt, int lb, cudaTextureObject_t texObject)
{

	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
	int p;

	float2 vterm, ploc;
	float vxterm, vyterm;

	// gtidx is the domain location in x for this thread
	if (gtidx < dx)
	{
		for (p = 0; p < lb; p++)
		{
			// fi is the domain location in y for this thread
			int fi = gtidy + p;

			if (fi < dy)
			{
				int fj = fi * pdx + gtidx;
				vterm = tex2D<float2>(texObject, (float)gtidx, (float)fi);
				ploc.x = (gtidx + 0.5f) - (dt * vterm.x * dx);
				ploc.y = (fi + 0.5f) - (dt * vterm.y * dy);
				vterm = tex2D<float2>(texObject, ploc.x, ploc.y);
				vxterm = vterm.x;
				vyterm = vterm.y;
				vx[fj] = vxterm;
				vy[fj] = vyterm;
			}
		}
	}
}

__global__ void diffuseProject_k(float2 *vx, float2 *vy, int dx, int dy, float dt, float visc, int lb)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
	int p;

	float2 xterm, yterm;

	// gtidx is the domain location in x for this thread
	if (gtidx < dx)
	{
		for (p = 0; p < lb; p++)
		{
			// fi is the domain location in y for this thread
			int fi = gtidy + p;

			if (fi < dy)
			{
				int fj = fi * dx + gtidx;
				xterm = vx[fj];
				yterm = vy[fj];

				// Compute the index of the wavenumber based on the
				// data order produced by a standard NN FFT.
				int iix = gtidx;
				int iiy = (fi > dy / 2) ? (fi - (dy)) : fi;

				// Velocity diffusion
				float kk = (float)(iix * iix + iiy * iiy); // k^2
				float diff = 1.f / (1.f + visc * dt * kk);
				xterm.x *= diff;
				xterm.y *= diff;
				yterm.x *= diff;
				yterm.y *= diff;

				// Velocity projection
				if (kk > 0.f)
				{
					float rkk = 1.f / kk;
					// (実数部分) Real portion of velocity projection
					float rkp = (iix * xterm.x + iiy * yterm.x);
					// (虚数部分) Imaginary portion of velocity projection
					float ikp = (iix * xterm.y + iiy * yterm.y);
					xterm.x -= rkk * rkp * iix;
					xterm.y -= rkk * ikp * iix;
					yterm.x -= rkk * rkp * iiy;
					yterm.y -= rkk * ikp * iiy;
				}

				vx[fj] = xterm;
				vy[fj] = yterm;
			}
		}
	}
}

__global__ void updateVelocity_k(float2 *v, float *vx, float *vy, int dx, int pdx, int dy, int lb, size_t pitch)
{

	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
	int p;

	float vxterm, vyterm;
	float2 nvterm;

	// gtidx is the domain location in x for this thread
	if (gtidx < dx)
	{
		for (p = 0; p < lb; p++)
		{
			// fi is the domain location in y for this thread
			int fi = gtidy + p;

			if (fi < dy)
			{
				int fjr = fi * pdx + gtidx;
				vxterm = vx[fjr];
				vyterm = vy[fjr];

				// Normalize the result of the inverse FFT
				float scale = 1.f / (dx * dy);
				nvterm.x = vxterm * scale;
				nvterm.y = vyterm * scale;

				float2 *fj = (float2 *)((char *)v + fi * pitch) + gtidx;
				*fj = nvterm;
			}
		} // If this thread is inside the domain in Y
	}	  // If this thread is inside the domain in X
}

__global__ void advectParticles_k(float2 *part, float2 *v, int dx, int dy, float dt, int lb, size_t pitch)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
	int p;

	// gtidx is the domain location in x for this thread
	float2 pterm, vterm;

	if (gtidx < dx)
	{
		for (p = 0; p < lb; p++)
		{
			// fi is the domain location in y for this thread
			int fi = gtidy + p;

			if (fi < dy)
			{
				int fj = fi * dx + gtidx;
				pterm = part[fj];

				// 粒子が速度場のどこに存在するか算出
				int xvi = ((int)(pterm.x * dx));
				int yvi = ((int)(pterm.y * dy));

				// 粒子の位置の速度を抽出
				vterm = *((float2 *)((char *)v + yvi * pitch) + xvi);

				// pterm.x += dt * vterm.x;
				// pterm.x *= (-2 * (0 < pterm.x) + 1); // 0 < pterm.x の時反発
				// vterm.x *= (-2 * (0 < pterm.x) + 1);
				// pterm.x -= ((pterm.x < 1) * 2 * (pterm.x - (int)pterm.x)); // pterm.x < 1 の時反発
				// vterm.x *= (-2 * (pterm.x < 1) + 1);

				// pterm.y += dt * vterm.y;
				// pterm.y *= (-2 * (0 < pterm.y) + 1); // 0 < pterm.x の時反発
				// vterm.y *= (-2 * (0 < pterm.y) + 1);
				// pterm.y -= ((pterm.y < 1) * 2 * (pterm.y - (int)pterm.y)); // pterm.x < 1 の時反発
				// vterm.y *= (-2 * (pterm.y < 1) + 1);

				pterm.x += dt * vterm.x;		  // 経過時間 * x方向速度
				pterm.x = pterm.x - (int)pterm.x; // 整数部分削除 (-1.0 ~ 1.0)
				pterm.x += 1.f;					  // (0.0 ~ 2.0)
				pterm.x = pterm.x - (int)pterm.x; // 整数部分削除 (0.0 ~ 1.0)
				pterm.y += dt * vterm.y;
				pterm.y = pterm.y - (int)pterm.y;
				pterm.y += 1.f;
				pterm.y = pterm.y - (int)pterm.y;

				part[fj] = pterm;
			}
		} // If this thread is inside the domain in Y
	}	  // If this thread is inside the domain in X
}
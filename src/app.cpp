#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>
#include "main.h"
#include "fluidsGL_kernels.h"

#include "app.h"

// const char *sSDKname = "fluidsGL";
// CUDA example code that implements the frequency space version of
// Jos Stam's paper 'Stable Fluids' in 2D. This application uses the
// CUDA FFT library (CUFFT) to perform velocity diffusion and to
// force non-divergence in the velocity field at each time step. It uses
// CUDA-OpenGL interoperability to update the particle field directly
// instead of doing a copy to system memory before drawing. Texture is
// used for automatic bilinear interpolation at the velocity advection step.

// CUFFT plan handle
cufftHandle planr2c;
cufftHandle planc2r;
static cData *vxfield = NULL;
static cData *vyfield = NULL;

cData *hvfield = NULL; // host velocity field
cData *dvfield = NULL; // device velocity field
static int wWidth = MAX(512, DIM);
static int wHeight = MAX(512, DIM);

static bool clicked = false;
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

// Particle data
GLuint vbo = 0;									// OpenGL vertex buffer object
struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange
static cData *particles = NULL;					// particle positions in host memory
static int lastx = 0, lasty = 0;

// Texture pitch
size_t tPitch = 0; // Now this is compatible with gcc in 64-bit

char *ref_file = NULL;
bool g_bQAAddTestForce = true;
int g_iFrameToCompare = 100;
int g_TotalErrors = 0;

bool g_bExitESC = false;

// CheckFBO/BackBuffer class objects
CheckRender *g_CheckRender = NULL;

extern "C" void addForces(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r);
extern "C" void advectVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy, float dt);
extern "C" void diffuseProject(cData *vx, cData *vy, int dx, int dy, float dt, float visc);
extern "C" void updateVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy);
extern "C" void advectParticles(GLuint vbo, cData *v, int dx, int dy, float dt);

void app::setup(void)
{
	// 実行時間取得用のタイマー
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);

	// Allocate and initialize host data
	hvfield = (cData *)malloc(sizeof(cData) * NUM_PRTICLES);
	memset(hvfield, 0, sizeof(cData) * NUM_PRTICLES);

	// Allocate and initialize device data
	cudaMallocPitch((void **)&dvfield, &tPitch, sizeof(cData) * DIM, DIM);

	// host to dicice
	cudaMemcpy(dvfield, hvfield, sizeof(cData) * NUM_PRTICLES, cudaMemcpyHostToDevice);

	// Temporary complex velocity field data
	cudaMalloc((void **)&vxfield, sizeof(cData) * PDS);
	cudaMalloc((void **)&vyfield, sizeof(cData) * PDS);

	setupTexture(DIM, DIM);

	// Create particle array
	particles = (cData *)malloc(sizeof(cData) * NUM_PRTICLES);
	memset(particles, 0, sizeof(cData) * NUM_PRTICLES);

	// 初期位置設定
	for (unsigned int y = 0; y < DIM; y++)
	{
		for (unsigned int x = 0; x < DIM; x++)
		{
			particles[y * DIM + x].x = (float)x / (float)DIM;
			particles[y * DIM + x].y = (float)y / (float)DIM;
		}
	}

	// Create CUFFT transform plan configuration
	checkCudaErrors(cufftPlan2d(&planr2c, DIM, DIM, CUFFT_R2C));
	checkCudaErrors(cufftPlan2d(&planc2r, DIM, DIM, CUFFT_C2R));

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cData) * NUM_PRTICLES, particles, GL_DYNAMIC_DRAW_ARB);

	GLint bsize; // Allocate and initialize host data
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);

	if (bsize != (sizeof(cData) * NUM_PRTICLES))
	{
		printf("Failed to initialize GL extensions.\n");
		exit(EXIT_FAILURE);
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
	getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
}

void app::update(void)
{
	sdkStartTimer(&timer);

	advectVelocity(dvfield, (float *)vxfield, (float *)vyfield, DIM, RPADW, DIM, DT);
	diffuseProject(vxfield, vyfield, CPADW, DIM, DT, VISCOSITY);
	updateVelocity(dvfield, (float *)vxfield, (float *)vyfield, DIM, RPADW, DIM);
	advectParticles(vbo, dvfield, DIM, DIM, DT);

	glutPostRedisplay(); // openGLに再描画を指示
}

void app::draw(void)
{
	defineViewMatrix();

	// render points from vertex buffer
	glClear(GL_COLOR_BUFFER_BIT);
	glColor4f(0, 1, 0, 0.5f);
	glPointSize(1);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnableClientState(GL_VERTEX_ARRAY);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 0, NULL);
	glDrawArrays(GL_POINTS, 0, NUM_PRTICLES);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisable(GL_TEXTURE_2D);

	// Finish timing before swap buffers to avoid refresh sync
	sdkStopTimer(&timer);
	glutSwapBuffers();

	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		sprintf(fps, "Cuda/GL Stable Fluids (%d x %d): %3.1f fps", DIM, DIM, ifps);
		glutSetWindowTitle(fps);
		fpsCount = 0;
		fpsLimit = (int)MAX(ifps, 1.f);
		sdkResetTimer(&timer);
	}
}

void app::defineViewMatrix(void)
{
	double eye[3] = {0.5, 0.5, 1.0};
	double center[3] = {0.5, 0.5, 0.0};
	double up[3] = {0.0, 1.0, 0.0};
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (double)wWidth / wHeight, 0.1, 100.0);
	glViewport(0, 0, wWidth, wHeight);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(eye[X], eye[Y], eye[Z], center[X], center[Y], center[Z], up[X], up[Y], up[Z]);
}

void app::keyPressed(unsigned char key)
{
	switch (key)
	{
	case 'r':
		memset(hvfield, 0, sizeof(cData) * NUM_PRTICLES);
		cudaMemcpy(dvfield, hvfield, sizeof(cData) * NUM_PRTICLES, cudaMemcpyHostToDevice);

		cudaGraphicsUnregisterResource(cuda_vbo_resource);

		getLastCudaError("cudaGraphicsUnregisterBuffer failed");

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(cData) * NUM_PRTICLES,
					 particles, GL_DYNAMIC_DRAW_ARB);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);

		getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
		break;

	default:
		break;
	}
}

void app::mousePressed(int x, int y, int button, int state)
{
	y = -y + wHeight;
	lastx = x;
	lasty = y;
	clicked = !clicked;
}

void app::mouseDragged(int x, int y)
{
	y = -y + wHeight;
	float fx = (lastx / (float)wWidth);
	float fy = (lasty / (float)wHeight);
	int nx = (int)(fx * DIM);
	int ny = (int)(fy * DIM);

	if (clicked && nx < DIM - FR && nx > FR - 1 && ny < DIM - FR && ny > FR - 1)
	{
		int ddx = x - lastx;
		int ddy = y - lasty;
		fx = ddx / (float)wWidth;
		fy = ddy / (float)wHeight;
		int spy = ny - FR;
		int spx = nx - FR;

		addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy, FR);

		lastx = x;
		lasty = y;
	}

	glutPostRedisplay(); // openGLに再描画を指示
}

void app::windowResize(int width, int height)
{
	wWidth = width;
	wHeight = height;

	defineViewMatrix();
	glutPostRedisplay(); // openGLに再描画を指示
}

void app::end(void)
{
	cudaGraphicsUnregisterResource(cuda_vbo_resource);

	deleteTexture();

	// Free all host and device resources
	free(hvfield);
	free(particles);
	cudaFree(dvfield);
	cudaFree(vxfield);
	cudaFree(vyfield);
	cufftDestroy(planr2c);
	cufftDestroy(planc2r);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &vbo);

	sdkDeleteTimer(&timer);
}

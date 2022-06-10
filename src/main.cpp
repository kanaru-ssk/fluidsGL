#include <helper_gl.h>
#include "main.h"
#include "app.h"
#include "fluidsGL_kernels.h"

void idle(void)
{
	app::update();
}

void display(void)
{
	app::draw();
}

void keyboard(unsigned char key, int x, int y)
{
	app::keyPressed(key);
}

void click(int button, int updown, int x, int y)
{
	app::mousePressed(x, y, button, updown);
}

void motion(int x, int y)
{
	app::mouseDragged(x, y);
}

void reshape(int width, int height)
{
	app::windowResize(width, height);
}

void cleanup(void)
{
	app::end();
}

void initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowPosition(INIT_X_POS, INIT_Y_POS);
	glutInitWindowSize(INIT_WIDTH, INIT_HEIGHT);
	glutCreateWindow("Jelly-Fish");
	glutIdleFunc(idle);
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(click);
	glutMotionFunc(motion);
	glutReshapeFunc(reshape);
	glutCloseFunc(cleanup);

	app::setup();
}

int main(int argc, char **argv)
{
	initGL(&argc, argv);
	glutMainLoop();
	return 0;
}

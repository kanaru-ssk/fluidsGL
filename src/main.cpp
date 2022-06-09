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

//	glewExperimental = GL_TRUE;
//	glewInit();

	app::setup();
}

int main(int argc, char **argv)
{
	initGL(&argc, argv);
        glutMainLoop();
	return 0;
}

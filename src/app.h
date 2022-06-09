#pragma once

#define X 0
#define Y 1
#define Z 2

#define DIV_W 32
#define DIV_H 32
#define NUM_POINTS DIV_W *DIV_H
#define NUM_INDEX 2 * (DIV_W + 1) * (DIV_H - 1)

#define PI 3.141592653589793
#define M 1.0
#define G 9.80665

class app
{
public:
	static void setup(void);
	static void update(void);
	static void draw(void);
	static void keyPressed(unsigned char key);
	static void mousePressed(int x, int y, int button, int state);
	static void mouseDragged(int x, int y);
	static void windowResize(int width, int height);
	static void end(void);
	static void defineViewMatrix(void);
};

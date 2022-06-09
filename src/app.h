#pragma once

#define X 0
#define Y 1
#define Z 2

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

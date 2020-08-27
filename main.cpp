#include "ApplicationManager.h"
#include "WindowConstants.h"


//callback methods created by Christof Rezk-Salama
//Künstliche Intelligenz Übung 2 - Musterlösung
ApplicationManager* app = 0;
void display(void) { app->display(); }
void reshape(int width, int height) { app->reshape(width, height); }
void keyboard(unsigned char key, int x, int y) { app->keyboard(key, x, y); }
void mouse(int button, int state, int x, int y) { app->mouseButtonEvent(button, state, x, y); }
void move(int x, int y) { app->mouseMoveEvent(x, y); }
void idle(void) { app->idle(); }
void timerEvent(int value) {
	glutPostRedisplay();
	glutTimerFunc(WC::REFRESH_DELAY, timerEvent, 0); 
}

int main(int argc, char **argv)
{
	app = new ApplicationManager();

	app->initGLContext(argc, argv);
	app->initCUDAContext(argc, argv);

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);
	glutTimerFunc(WC::REFRESH_DELAY, timerEvent, 0);

	glutMouseFunc(mouse);
	glutMotionFunc(move);

	app->run(argc, argv);

	delete app;

	exit(EXIT_SUCCESS);
}
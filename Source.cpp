#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ImageController.h"

using namespace cv;
using namespace std;

int main(int argc, const char** argv) {
	imgctrl::Image image = imgctrl::Image::load("C://1.jpg");
	imgctrl::ImageController imgController;
	//image = imgController.getGrayScale(image);
	image = imgController.getBinarization(image);
	namedWindow("Test1", WINDOW_AUTOSIZE);
	imshow("Test1", Mat(image));

	waitKey(0);

	return 0;
}
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ImageController.h"

using namespace cv;
using namespace std;

bool isTriangle(const imgctrl::Image& image);
int getNumOfEdges(const imgctrl::Image& image);

int main(int argc, const char** argv) {
	imgctrl::Image image = imgctrl::Image::load("C://triangle.jpg");
	//imgctrl::Image image = imgctrl::Image::load("C://star.jpg");
	cout << (isTriangle(image)? "Triangle" : "Star") << endl;
	return 0;
}

bool isTriangle(const imgctrl::Image& image) {
	return getNumOfEdges(image) <= 4;
}

int getNumOfEdges(const imgctrl::Image& image) {
	imgctrl::ImageController imgController;
	vector<vector<double> > filters[] = {		// Roberts mask
		{ {0, 0, -1}, {0, 1, 0}, {0, 0, 0} },
		{ {-1, 0, 0}, {0, 1, 0}, {0, 0, 0} },
		{ {0, 0, 0}, {0, 1, 0}, {-1, 0, 0} },
		{ {0, 0, 0}, {0, 1, 0}, {0, 0, -1} },
	};

	// Use multiple mask and composite to detect all direction
	imgctrl::Image maskedImage(image.getSize());
	for (int i = 0; i < 4; i++){
		auto tmpImage = imgController.getConvolution(image, filters[i]);
		maskedImage = imgController.getCompositiion(maskedImage, tmpImage);
	}

	// Detect lines
	auto lines = imgController.getHoughLine(maskedImage);

	// for debug, print data of lines
	/*
	for (auto line : lines) {
		cout << line.rho << " " << line.ang << endl;
	}
	*/

	// Draw lines and Display
	/*
	auto displayImage = imgController.getLinedImage(image, lines);
	namedWindow("Display", WINDOW_AUTOSIZE);
	imshow("Display", Mat(displayImage));
	waitKey(0);
	*/

	return lines.size();
}


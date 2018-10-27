#include <iostream>
#include <chrono>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ImageController.h"

using namespace cv;
using namespace std;
static int numOfDisplay=0;

bool isTriangle(const imgctrl::Image& image);
int getNumOfEdges(const imgctrl::Image& image);

int main(int argc, const char** argv) {
	auto start = std::chrono::system_clock::now();
	imgctrl::Image images[] = {
		imgctrl::Image::load("C://1.jpg"),
		imgctrl::Image::load("C://2.jpg"),
		imgctrl::Image::load("C://3.jpg"),
		imgctrl::Image::load("C://4.jpg"),
	};
	//imgctrl::Image image = imgctrl::Image::load("C://star.jpg");
	for (int i = 0; i < 4; i++) {
		images[i].resize({ 1000, 1000 });
		cout << i+1 << ".jpg : " << (isTriangle(images[i]) ? "Triangle" : "Star") << endl;
	}

	auto end = std::chrono::system_clock::now();
	chrono::duration<double> defaultSec = end - start;
	cout << defaultSec.count() << endl;
	waitKey(0);
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
		maskedImage = imgController.getComposition(maskedImage, tmpImage);
	}

	// Detect lines
	auto lines = imgController.getHoughLine(maskedImage);

	// Draw lines and Display
	auto displayImage = image;//imgController.getLinedImage(image, lines);
	numOfDisplay++;
	string windowName = std::to_string(numOfDisplay) + ".jpg";
	namedWindow(windowName, WINDOW_NORMAL);
	//namedWindow(windowName, WINDOW_AUTOSIZE);
	imshow(windowName, Mat(displayImage));

	return lines.size();
}


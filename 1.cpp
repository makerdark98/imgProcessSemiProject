#include <iostream>
#include <chrono>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ImageController.h"

using namespace cv;
using namespace std;

void displayImage(const string& windowName, const imgctrl::Image& image);
bool isTriangle(const imgctrl::Image& image);
int getNumOfEdges(const imgctrl::Image& image);

int main(int argc, const char** argv) {
	auto start = std::chrono::system_clock::now();
	string answerSheet[4];
	imgctrl::Image images[] = {
		imgctrl::Image::load("C://1.jpg"), imgctrl::Image::load("C://2.jpg"),
		imgctrl::Image::load("C://3.jpg"), imgctrl::Image::load("C://4.jpg"),
	};
	for (int i = 0; i < 4; i++) {
		images[i].resize({ 1000, 1000 });
		answerSheet[i] = isTriangle(images[i]) ? "Triangle" : "Star";
	}

	auto end = std::chrono::system_clock::now();
	chrono::duration<double> determiningTime = end - start;
	cout << determiningTime.count() << endl;
	start = end;

	for (int i = 0; i < 4; i++) {
		displayImage(std::to_string(i + 1) + ".jpg :" + answerSheet[i], images[i]);
	}

	end = std::chrono::system_clock::now();
	chrono::duration<double> displayTime = end - start;
	cout << displayTime.count() << endl;

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

	return lines.size();
}

void displayImage(const string& windowName, const imgctrl::Image& image) {
	namedWindow(windowName, WINDOW_NORMAL);
	imshow(windowName, Mat(image));
}

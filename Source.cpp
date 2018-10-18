#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ImageController.h"

using namespace cv;
using namespace std;

void processHoughLine(string filename) {
	imgctrl::Image image = imgctrl::Image::load(filename);
	filename = filename + "Line";
	imgctrl::ImageController imgController;
	vector<vector<double> > filter = {
		{0, 0, -1},
		{0, 1, 0},
		{0, 0, 0}
	};
	image = imgController.getConvolution(image, filter);
	auto lines = imgController.getHoughLine(image);
	cout << filename << endl;
	for (auto line : lines) {
		cout << line.rho << " " << line.ang << endl;
	}
	namedWindow(filename, WINDOW_AUTOSIZE);
	imshow(filename, Mat(image));
}

void processCorner(string filename) {
	imgctrl::Image image = imgctrl::Image::load(filename);
	filename = filename + "Corner";
	imgctrl::ImageController imgController;
	imgController.setThreshold(20000);
	auto corners = imgController.getHarrisCorner(image);
	image = imgController.getMarkedImage(image, corners);
	namedWindow(filename, WINDOW_AUTOSIZE);
	imshow(filename, Mat(image));

}

int main(int argc, const char** argv) {
	//processHoughLine("C://triangle.jpg");
	//processHoughLine("C://star.jpg");
	processCorner("C://triangle.jpg");
	processCorner("C://star.jpg");
	waitKey(0);
	return 0;
}
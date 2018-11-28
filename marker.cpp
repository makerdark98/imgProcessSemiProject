#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ImageController.h"

using namespace std;

class QuadPoint {
public:
	vector<imgctrl::Point> quad;

	QuadPoint(const imgctrl::Point& topLeft, const imgctrl::Point& topRight, const imgctrl::Point& bottomRight, const imgctrl::Point& bottomLeft) {
		quad.push_back(topLeft);
		quad.push_back(topRight);
		quad.push_back(bottomRight);
		quad.push_back(bottomLeft);
	}
};

cv::Mat getRawImageToMarker(const imgctrl::Image& image, const QuadPoint& markerPosition);

int main(int argc, char** argv) {
	// if you use function that use threshold value, must set threshold plz
	imgctrl::ImageController imgController;

	// See constructor parameter name. order tl, tr, br, bl 
	QuadPoint inputQuad( {104., 82.}, {207., 108.}, {182., 222.}, {84., 194.});

	auto cvImage = cv::imread("C:\\2.jpg", CV_LOAD_IMAGE_COLOR);
	auto cvResult = getRawImageToMarker(cvImage, inputQuad); // cvImage will be clone into imgctrl::Image object
	cv::namedWindow("OPENCV", cv::WINDOW_NORMAL);
	cv::imshow("OPENCV", cvResult);

	// recommend
	imgctrl::Image image = imgctrl::Image::load("C:\\2.jpg");
	auto result = getRawImageToMarker(image, inputQuad);
	cv::namedWindow("Custom", cv::WINDOW_NORMAL);
	cv::imshow("Custom", result);
	cv::waitKey(0);

	return 0;
}

cv::Mat getRawImageToMarker(const imgctrl::Image & image, const QuadPoint & markerPosition)
{
	QuadPoint outputQuad( { 0., 0. }, { 140., 0. }, { 140., 140. }, { 0., 140. });

	// inverse Perspective, Original Order is input, output
	auto M = imgctrl::getPerspectiveMatrix(outputQuad.quad, markerPosition.quad);

	// Fix size
	imgctrl::Image result({ 140, 140 });

	// warping
	for (size_t i = 0; i < result.getWidth(); i++) {
		for (size_t j = 0; j < result.getHeight(); j++) {
			imgctrl::Matrix ori(3, 1);
			ori.data[0][0] = i;
			ori.data[1][0] = j;
			ori.data[2][0] = 1;

			auto trans = M * ori;
			
			double w = trans.data[2][0];
			double x = trans.data[0][0] / w;
			double y = trans.data[1][0] / w;
			if (x < 0 || y < 0 || x >= image.getWidth() || y >= image.getHeight()) continue; // Out of range

			result[i][j] = image[x][y];
		}
	}

	return result;
}

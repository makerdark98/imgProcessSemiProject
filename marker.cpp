#include <iostream>
#include <queue>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ImageController.h"

using namespace std;

class QuadPoint;
imgctrl::Image getInversePerspective(const imgctrl::Image& image, const QuadPoint& markerPosition);
imgctrl::Point getIntersection(const imgctrl::LineParam& line0, const imgctrl::LineParam& line1);
imgctrl::Image getMarkerImage(const imgctrl::Image& image);
void sampleExtractMarker();

int main(int argc, char** argv) {
	imgctrl::ImageController imgController;
	imgctrl::Image image = imgctrl::Image::load("c:\\2.jpg");
	cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
	cv::imshow("Original", cv::Mat(image));
	cv::namedWindow("Marker", cv::WINDOW_AUTOSIZE);
	cv::imshow("Marker", cv::Mat(getMarkerImage(image)));
	cv::waitKey(0);
	return 0;
}

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

void sampleExtractMarker() {
	// if you use function that use threshold value, must set threshold plz
	imgctrl::ImageController imgController;

	// See constructor parameter name. order tl, tr, br, bl 
	QuadPoint inputQuad( {104., 82.}, {207., 108.}, {182., 222.}, {84., 194.});

	auto cvImage = cv::imread("C:\\2.jpg", CV_LOAD_IMAGE_COLOR);
	auto cvResult = getInversePerspective(cvImage, inputQuad); // cvImage will be clone into imgctrl::Image object
	cv::namedWindow("OPENCV", cv::WINDOW_AUTOSIZE);
	cv::imshow("OPENCV", cv::Mat(cvResult));

	// recommend
	imgctrl::Image image = imgctrl::Image::load("C:\\2.jpg");
	auto result = getInversePerspective(image, inputQuad);
	cv::namedWindow("Custom", cv::WINDOW_AUTOSIZE);
	cv::imshow("Custom", cv::Mat(result));
	cv::waitKey(0);
}

imgctrl::Image getInversePerspective(const imgctrl::Image & image, const QuadPoint & markerPosition)
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

imgctrl::Point getIntersection(const imgctrl::LineParam & line0, const imgctrl::LineParam & line1)
{
	const double theta0 = line0.ang*M_PI/180;
	const double theta1 = line1.ang*M_PI/180;
	const double& r0 = line0.rho;
	const double& r1 = line1.rho;

	double det = cos(theta0)*sin(theta1) - cos(theta1)*sin(theta0);
	double x = sin(theta1)*r0 - sin(theta0)*r1;
	double y = -cos(theta1)*r0 + cos(theta0)*r1;
	x /= det;
	y /= det;
	return imgctrl::Point({ y, x });
}

imgctrl::Image getMarkerImage(const imgctrl::Image & image)
{
	imgctrl::ImageController imgController;
	vector<vector<double> > filters[] = {		// Roberts mask
		{ {0, 0, -1}, {0, 1, 0}, {0, 0, 0} },
		{ {-1, 0, 0}, {0, 1, 0}, {0, 0, 0} },
		{ {0, 0, 0}, {0, 1, 0}, {-1, 0, 0} },
		{ {0, 0, 0}, {0, 1, 0}, {0, 0, -1} },
	};

	// Use multiple mask and composite to detect all direction
	imgctrl::Image maskedImage(image.getSize());
	imgController.setThreshold(30);
	auto binaryImage = imgController.getBinarization(image);

	// TO display
	cv::namedWindow("binary", cv::WINDOW_AUTOSIZE);
	cv::imshow("binary", cv::Mat(binaryImage));

	for (int i = 0; i < 4; i++){
		auto tmpImage = imgController.getConvolution(binaryImage, filters[i]);
		maskedImage = imgController.getComposition(maskedImage, tmpImage);
	}

	// TO display
	cv::namedWindow("masked", cv::WINDOW_AUTOSIZE);
	cv::imshow("masked", cv::Mat(maskedImage));

	// Detect lines
	imgController.setThreshold(80);
	auto lines = imgController.getHoughLine(maskedImage);

	// TO display
	maskedImage = imgController.getLinedImage(maskedImage, lines);
	cv::namedWindow("line", cv::WINDOW_AUTOSIZE);
	cv::imshow("line", cv::Mat(maskedImage));

	vector<imgctrl::Point> corners;
	for (size_t i = 0; i < lines.size(); i++) {
		for (size_t j = i+1; j < lines.size(); j++) {
			if (fabs(lines[i].ang - lines[j].ang) <= 2.) continue;
			auto interPoint = getIntersection(lines[i], lines[j]);
			if (interPoint.x < 0 || interPoint.x >= image.getWidth()
				|| interPoint.y < 0 || interPoint.y >= image.getHeight())
				continue;
			corners.push_back(interPoint);
		}
	}

	//TODO : sort clockwise
	if (corners.size() == 4) {
		for (int i = 0; i < 4; i++) cout << corners[i].x << " " << corners[i].y << endl;
		QuadPoint markerPosition(corners[2], corners[3], corners[1], corners[0]);
		return getInversePerspective(image, markerPosition);
	}
	return imgctrl::Image({ 0,0 });
}

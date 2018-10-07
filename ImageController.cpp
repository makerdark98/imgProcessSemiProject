#include "ImageController.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"


imgctrl::ImageController::ImageController()
{
}

imgctrl::ImageController::~ImageController()
{
}

BYTE imgctrl::ImageController::getThreshold() const
{
	return threshold;
}

void imgctrl::ImageController::setThreshold(const BYTE & threshold)
{
	this->threshold = threshold;
}

imgctrl::Image imgctrl::ImageController::getBinarization(const Image & original) const
{
	Image result = getGrayScale(original);
	std::pair<size_t, size_t> size = result.getSize();
	for (size_t i = 0; i < size.first; i++) {
		for (size_t j = 0; j < size.second; j++) {
			BYTE newColor = this->binaryWhite;
			if (result.m_image[i][j].getRed() < this->threshold) {
				newColor = this->binaryBlack;
			}
			result.m_image[i][j].setRed(newColor);
			result.m_image[i][j].setGreen(newColor);
			result.m_image[i][j].setBlue(newColor);
		}
	}
	return result;
}

imgctrl::Image imgctrl::ImageController::getGrayScale(const Image & original) const
{
	Image result = original;
	std::pair<size_t, size_t> size = result.getSize();
	for (size_t i = 0; i < size.first; i++) {
		for (size_t j = 0; j < size.second; j++) {
			float r, g, b;
			r = (float)result.m_image[i][j].getRed();
			g = (float)result.m_image[i][j].getGreen();
			b = (float)result.m_image[i][j].getBlue();
			float gray = min(255.f, max(0.f, 0.2126f * r + 0.7152f * g + 0.0722f * b));
			result.m_image[i][j].setRed((BYTE)gray);
			result.m_image[i][j].setGreen((BYTE)gray);
			result.m_image[i][j].setBlue((BYTE)gray);
		}
	}
	return result;
}

std::vector<std::pair<unsigned int,unsigned int>> imgctrl::ImageController::getHarrisCorner(const Image & image) const
{
	std::vector<std::pair<unsigned int,unsigned int> > result;
	double tx, ty;
	std::pair<size_t, size_t> size = image.getSize();
	std::vector<std::vector<double> > dx2(size.first, std::vector<double>(size.second, 0));
	std::vector<std::vector<double> > dy2(size.first, std::vector<double>(size.second, 0));
	std::vector<std::vector<double> > dxy(size.first, std::vector<double>(size.second, 0));

	for (size_t i = 1; i < size.first-1; i++) {
		for (size_t j = 1; j < size.second-1; j++) {
			tx = ((double)image.m_image[i - 1][j + 1].getRed() + (double)image.m_image[i][j + 1].getRed() + (double)image.m_image[i + 1][j + 1].getRed()
				- (double)image.m_image[i - 1][j - 1].getRed() - (double)image.m_image[i][j - 1].getRed() - (double)image.m_image[i + 1][j - 1].getRed()) / 6.0;
			ty = ((double)image.m_image[i + 1][j - 1].getRed() + (double)image.m_image[i + 1][j].getRed() + (double)image.m_image[i + 1][j + 1].getRed()
				- (double)image.m_image[i - 1][j - 1].getRed() + (double)image.m_image[i - 1][j].getRed() - (double)image.m_image[i - 1][j + 1].getRed()) / 6.0;

			dx2[i][j] = tx*tx;
			dy2[i][j] = ty*ty;
			dxy[i][j] = tx*ty;
		}
	}

	std::vector<std::vector<double> > gdx2(size.first, std::vector<double>(size.second, 0));
	std::vector<std::vector<double> > gdy2(size.first, std::vector<double>(size.second, 0));
	std::vector<std::vector<double> > gdxy(size.first, std::vector<double>(size.second, 0));
	double g[5][5] = {
		{1, 4, 6 ,4, 1},
		{4, 16, 24, 16, 4},
		{6, 24, 36, 24, 6},
		{4, 16, 24, 16, 4},
		{1, 4, 6 ,4, 1}
	};

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			g[i][j] /= 256.;
		}
	}

	double tx2, ty2, txy;
	for (size_t i = 2; i < size.first - 2; i++) {
		for (size_t j = 2; j < size.second - 2; j++) {
			tx2 = 0;
			ty2 = 0;
			txy = 0;
			for (int x = 0; x < 5; x++) {
				for (int y = 0; y < 5; y++) {
					tx2 += (dx2[i + x - 2][j + y - 2] * g[x][y]);
					ty2 += (dy2[i + x - 2][j + y - 2] * g[x][y]);
					txy += (dxy[i + x - 2][j + y - 2] * g[x][y]);
				}
			}
			gdx2[i][j] = tx2;
			gdy2[i][j] = ty2;
			gdxy[i][j] = txy;
		}
	}

	std::vector<std::vector<double> > crf(size.first, std::vector<double>(size.second, 0));

	double k = 0.04;
	for (size_t i = 2; i < size.first - 2; i++) {
		for (size_t j = 2; j < size.second - 2; j++) {
			crf[i][j] = (gdx2[i][j] * gdy2[i][j] - gdxy[i][j] * gdxy[i][j]) - k*(gdx2[i][j] + gdy2[i][j])*(gdx2[i][j] + gdy2[i][j]);
		}
	}

	for (size_t i = 2; i < size.first - 2; i++) {
		for (size_t j = 2; j < size.second - 2; j++) {
			if (crf[i][j] > threshold &&
				crf[i][j] > crf[i - 1][j] && crf[i][j] > crf[i - 1][j + 1] &&
				crf[i][j] > crf[i][j + 1] && crf[i][j] > crf[i + 1][j + 1] &&
				crf[i][j] > crf[i + 1][j] && crf[i][j] > crf[i + 1][j - 1] &&
				crf[i][j] > crf[i][j - 1] && crf[i][j] > crf[i - 1][j - 1]){
				result.push_back({ i,j });
			}
		}
	}

	return result;
}

imgctrl::Image imgctrl::ImageController::getMarkedImage(const Image & original, std::vector<std::pair<unsigned int, unsigned int>>& markPositions, unsigned int markSize)
{
	Image result = original;
	std::pair<size_t, size_t> size = result.getSize();
	for (std::pair<unsigned int, unsigned int> markPosition : markPositions) {
		for (unsigned int dx = 0; dx < markSize && markPosition.first + dx < size.first; dx++) {
			for (unsigned int dy = 0; dy < markSize && markPosition.second + dy < size.first; dy++) {
				result.m_image[markPosition.first + dx][markPosition.second + dy].setRed(255);
				result.m_image[markPosition.first + dx][markPosition.second + dy].setGreen(0);
				result.m_image[markPosition.first + dx][markPosition.second + dy].setBlue(0);
			}
		}
	}
	return result;
}

imgctrl::Image::Image(std::vector<std::vector<Color>> image)
{
	m_image = image;
}

imgctrl::Image::~Image()
{
}


imgctrl::Image imgctrl::Image::load(std::string filename)
{
	cv::Mat cvImage = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	int type = cvImage.type();
	cv::Size size = cvImage.size();
	std::vector< std::vector<Color> > data(size.width, std::vector<Color>(size.height));
	for (int i = 0; i < size.width; i++) {
		for (int j = 0; j < size.height; j++) {
			cv::Vec3b& currentColor = cvImage.at<cv::Vec3b>(cv::Point(i, j));
			data[i][j].setRed(currentColor[2]);
			data[i][j].setGreen(currentColor[1]);
			data[i][j].setBlue(currentColor[0]);
		}
	}
	return Image(data);
}

void imgctrl::Image::save(std::string filename) const
{
}

size_t imgctrl::Image::getHeight() const
{
	return m_image.size();
}

size_t imgctrl::Image::getWidth() const
{
	if (m_image.size() == 0) return 0;
	return m_image[0].size();
}

std::pair<size_t, size_t> imgctrl::Image::getSize() const
{
	return std::pair<size_t, size_t>({getHeight(), getWidth()});
}

imgctrl::Image::operator cv::Mat() const
{
	std::pair<size_t, size_t> size = this->getSize();
	cv::Size resultSize(size.first, size.second);
	cv::Mat result(resultSize, CV_8UC3);
	for (size_t i = 0; i < size.first; i++) {
		for (size_t j = 0; j < size.second; j++) {
			cv::Vec3b& currentColor = result.at<cv::Vec3b>(cv::Point(i, j));
			currentColor[2] = m_image[i][j].getRed();
			currentColor[1] = m_image[i][j].getGreen();
			currentColor[0] = m_image[i][j].getBlue();
		}
	}
	return result;
}

imgctrl::Color::Color() {
	setRed(0);
	setGreen(0);
	setBlue(0);
}

imgctrl::Color::Color(BYTE r, BYTE g, BYTE b)
{
	setRed(r);
	setGreen(g);
	setBlue(b);
}

imgctrl::Color::~Color()
{
}

void imgctrl::Color::setRed(BYTE r)
{
	m_color[kRedIdx] = r;
}

void imgctrl::Color::setGreen(BYTE g)
{
	m_color[kGreenIdx] = g;
}

void imgctrl::Color::setBlue(BYTE b)
{
	m_color[kBlueIdx] = b;
}

BYTE imgctrl::Color::getRed() const
{
	return m_color[kRedIdx];
}

BYTE imgctrl::Color::getGreen() const
{
	return m_color[kGreenIdx];
}

BYTE imgctrl::Color::getBlue() const
{
	return m_color[kBlueIdx];
}

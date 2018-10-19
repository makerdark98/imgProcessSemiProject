#include "ImageController.h"
#include "opencv2/imgproc/imgproc.hpp"
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
	return m_threshold;
}

void imgctrl::ImageController::setThreshold(const int & threshold)
{
	this->m_threshold = threshold;
}

imgctrl::Image imgctrl::ImageController::getBinarization(const Image & original) const
{
	Image result = getGrayScale(original);
	const std::pair<size_t, size_t> size = result.getSize();
	for (unsigned int i = 0; i < size.first; i++) {
		for (unsigned int j = 0; j < size.second; j++) {
			BYTE newColor = this->kWhiteBinary;
			if (result[i][j].getRed() < this->m_threshold) {
				newColor = this->kBlackBinary;
			}
			result[i][j].setColor(newColor, newColor, newColor);
		}
	}
	return result;
}

imgctrl::Image imgctrl::ImageController::getBlur(const Image & original) const
{
	const std::vector<std::vector<double> > filter = {
		{1./9, 1./9, 1./9},
		{1./9, 1./9, 1./9},
		{1./9, 1./9, 1./9}
	};
	return getConvolution(original, filter);
}

imgctrl::Image imgctrl::ImageController::getSharpening(const Image & original) const
{
	const std::vector<std::vector<double> > filter = {
		{-1, -1, -1},
		{-1, 9, -1},
		{-1, -1, -1}
	};
	return getConvolution(original, filter);
}

imgctrl::Image imgctrl::ImageController::getGrayScale(const Image & original) const
{
	Image result = original;
	const std::pair<size_t, size_t> size = result.getSize();

	for (unsigned int i = 0; i < size.first; i++) {
		for (unsigned int j = 0; j < size.second; j++) {
			float r, g, b;
			r = (float)result[i][j].getRed();
			g = (float)result[i][j].getGreen();
			b = (float)result[i][j].getBlue();

			float gray = std::min(255.f, std::max(0.f, 0.2126f * r + 0.7152f * g + 0.0722f * b));
			result[i][j].setColor(gray, gray, gray);
		}
	}

	return result;
}

imgctrl::Image imgctrl::ImageController::getConvolution(const Image & original, const std::vector<std::vector<double>>& filter) const
{
	// Because I usually make a mistake about c++ basic syntax, add assert statement.
	assert(filter.size() != 0);

	const std::pair<size_t, size_t> imageSize = original.getSize();
	Image result(imageSize);

	const std::pair<size_t, size_t> filterSize = { filter.size(), filter[0].size() };
	unsigned int halfX = filterSize.first / 2;
	unsigned int halfY = filterSize.second / 2;

	for (unsigned int i = halfX; i < imageSize.first-halfX; i++) {
		for (unsigned int j = halfY; j < imageSize.second-halfY; j++) {
			double r = 0.;
			double g = 0.; 
			double b = 0.;

			for (unsigned int x = 0; x < filterSize.first; x++) {
				for (unsigned int y = 0; y < filterSize.second; y++) {
					r += filter[x][y] * original[i-halfX+x][j-halfY+y].getRed();
					g += filter[x][y] * original[i-halfX+x][j-halfY+y].getGreen();
					b += filter[x][y] * original[i-halfX+x][j-halfY+y].getBlue();
				}
			}

			// Caution! Overflow Check
			r = std::min(255., std::max(0., r));
			g = std::min(255., std::max(0., g));
			b = std::min(255., std::max(0., b));

			result[i][j].setColor(r, g, b);
		}
	}
	return result;
}

std::vector<std::pair<unsigned int,unsigned int>> imgctrl::ImageController::getHarrisCorner(const Image & image) const
{
	/* 
	Please See HarrisCorner Algorithm 
	Caution! It does not always detect corner
	In this assignment, I have not use this algorithm 
	*/

	std::vector<std::pair<unsigned int,unsigned int> > result;
	double tx, ty;
	const std::pair<size_t, size_t> imgSize = image.getSize();

	std::vector<std::vector<double> > dx2(imgSize.first, std::vector<double>(imgSize.second, 0));
	std::vector<std::vector<double> > dy2(imgSize.first, std::vector<double>(imgSize.second, 0));
	std::vector<std::vector<double> > dxy(imgSize.first, std::vector<double>(imgSize.second, 0));

	for (unsigned int i = 1; i < imgSize.first-1; i++) {
		for (unsigned int j = 1; j < imgSize.second-1; j++) {
			tx = ((double)image[i - 1][j + 1].getRed() + (double)image[i][j + 1].getRed() + (double)image[i + 1][j + 1].getRed()
				- (double)image[i - 1][j - 1].getRed() - (double)image[i][j - 1].getRed() - (double)image[i + 1][j - 1].getRed()) / 6.0;
			ty = ((double)image[i + 1][j - 1].getRed() + (double)image[i + 1][j].getRed() + (double)image[i + 1][j + 1].getRed()
				- (double)image[i - 1][j - 1].getRed() + (double)image[i - 1][j].getRed() - (double)image[i - 1][j + 1].getRed()) / 6.0;

			dx2[i][j] = tx*tx;
			dy2[i][j] = ty*ty;
			dxy[i][j] = tx*ty;
		}
	}

	/* 
	Gaussian filtering (Blur) dx2, dy2, dxy
	*/
	std::vector<std::vector<double> > gdx2(imgSize.first, std::vector<double>(imgSize.second, 0));
	std::vector<std::vector<double> > gdy2(imgSize.first, std::vector<double>(imgSize.second, 0));
	std::vector<std::vector<double> > gdxy(imgSize.first, std::vector<double>(imgSize.second, 0));
	double g[5][5] = {
		{1, 4, 6 ,4, 1},
		{4, 16, 24, 16, 4},
		{6, 24, 36, 24, 6},
		{4, 16, 24, 16, 4},
		{1, 4, 6 ,4, 1}
	};

	for (unsigned int i = 0; i < 5; i++) {
		for (unsigned int j = 0; j < 5; j++) {
			g[i][j] /= 256.;
		}
	}

	double tx2, ty2, txy;
	for (unsigned int i = 2; i < imgSize.first - 2; i++) {
		for (unsigned int j = 2; j < imgSize.second - 2; j++) {
			tx2 = 0;
			ty2 = 0;
			txy = 0;
			for (unsigned int x = 0; x < 5; x++) {
				for (unsigned int y = 0; y < 5; y++) {
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

	std::vector<std::vector<double> > crf(imgSize.first, std::vector<double>(imgSize.second, 0));

	/* 
	Use Taylor Series to simplify calculation
	If you want to know more, See Harris Corner Response Function
	*/
	const double k = 0.04;
	for (unsigned int i = 2; i < imgSize.first - 2; i++) {
		for (unsigned int j = 2; j < imgSize.second - 2; j++) {
			crf[i][j] = (gdx2[i][j] * gdy2[i][j] - gdxy[i][j] * gdxy[i][j]) - k*(gdx2[i][j] + gdy2[i][j])*(gdx2[i][j] + gdy2[i][j]);
		}
	}

	auto isLocalMaximum = [&crf, &result, &imgSize](const size_t& x, const size_t& y, const int& range=50)->bool{
		for (int i = -range; i <= range; i++) {
			for (int j = -range; j <= range; j++) {
				if (i + x < 0 || i + x >= imgSize.first || j + y < 0 || j + y >= imgSize.second) continue;
				if (crf[i + x][j + y] - crf[x][y] >= FLT_EPSILON) return false;
			}
		}
		return true;
	};

	for (unsigned int i = 0; i < imgSize.first; i++) {
		for (unsigned int j = 0; j < imgSize.second; j++) {
			if (crf[i][j] > m_threshold && isLocalMaximum(i,j)){
				result.push_back({ i,j });
			}
		}
	}

	return result;
}

std::vector<imgctrl::LineParam> imgctrl::ImageController::getHoughLine(const Image & image) const
{
	/* 
	Caution! It does not detect all line. 
	If you want to increase prediction rate, must refine codes to find local minimum 
	*/

	/* HoughLine Algorithm is more better performance in Binary Image*/
	Image grayImage = getBinarization(image);

	std::vector<LineParam> result;
	const std::pair<size_t, size_t> size = image.getSize();

	int num_rho = (int)(sqrt((double)size.first*size.first + size.second*size.second) * 2);
	int num_ang = 360;

	std::vector<double> tsin(num_ang), tcos(num_ang);

	for (int i = 0; i < num_ang; i++) {
		tsin[i] = (double)sin(i*M_PI / num_ang);
		tcos[i] = (double)cos(i*M_PI / num_ang);
	}

	std::vector<std::vector<int> > accumulation(num_rho, std::vector<int>(num_ang, 0));

	int m, n;
	for (unsigned int i = 0; i < size.first; i++) {
		for (unsigned int j = 0; j < size.second; j++) {
			if (grayImage[i][j].getRed() > m_threshold){ 
				for (int n = 0; n < num_ang; n++) {
					m = (int)floor(i*tsin[n] + j*tcos[n] + 0.5);
					m += (num_rho / 2);

					accumulation[m][n]++;
				}
			}
		}
	}

	// I don't know where to place this function.... So just use lambda
	auto isLocalMaximum = [&accumulation, &result, &num_rho, &num_ang](const size_t& rho, const size_t& ang, const int& rhoRange=10, const int& angRange=2)->bool{
		for (int i = -rhoRange; i <= rhoRange; i++) {
			for (int j = -angRange; j <= angRange; j++) {
				if (i + rho < 0 || i + rho >= num_rho || j + ang < 0 || j + ang >= num_ang ) continue;
				if (i == 0 && j == 0) continue;
				if (accumulation[i + rho][j + ang] - accumulation[rho][ang] >= FLT_EPSILON) return false;
			}
		}
		return true;
	};

	for (m = 1; m < num_rho-1; m++) {
		for (n = 0; n < num_ang-1; n++) {
			if (accumulation[m][n] > m_threshold &&
				isLocalMaximum(m, n) ) {
				result.push_back({ (double)m - (num_rho / 2), (double)n*180.0 / num_ang });
			}
		}
	}

	return result;
}

imgctrl::Image imgctrl::ImageController::getMarkedImage(const Image & original, const std::vector<std::pair<unsigned int, unsigned int>>& markPositions, const unsigned int& markSize)
{
	Image result = original;
	const std::pair<size_t, size_t> imgSize = result.getSize();

	for (std::pair<unsigned int, unsigned int> markPosition : markPositions) {
		for (unsigned int dx = 0; dx < markSize && markPosition.first + dx < imgSize.first; dx++) {
			for (unsigned int dy = 0; dy < markSize && markPosition.second + dy < imgSize.first; dy++) {
				result[markPosition.first + dx][markPosition.second + dy].setColor(255, 0, 0);
			}
		}
	}
	return result;
}

imgctrl::Image imgctrl::ImageController::getCompositiion(const Image & firstImage, const Image & secondImage) const
{
	auto firstSize = firstImage.getSize();
	auto secondSize = secondImage.getSize();

	assert(firstSize.first == secondSize.first);
	assert(firstSize.second == secondSize.second);

	Image result(firstSize);
	for (unsigned int x = 0; x < firstSize.first; x++) {
		for (unsigned int y = 0; y < firstSize.second; y++) {
			int r, g, b;
			r = firstImage[x][y].getRed() + secondImage[x][y].getRed();
			g = firstImage[x][y].getGreen() + secondImage[x][y].getGreen();
			b = firstImage[x][y].getBlue() + secondImage[x][y].getBlue();
			r = std::min(255, std::max(0, r));
			g = std::min(255, std::max(0, g));
			b = std::min(255, std::max(0, b));
			result[x][y].setColor(r, g, b);
		}
	}
	return result;

}

imgctrl::Image imgctrl::ImageController::getLinedImage(const Image & original, const std::vector<imgctrl::LineParam>& lines) const
{
	Image result = original;
	std::pair<size_t, size_t> imgSize = result.getSize();
	for (auto line : lines) {
		// Generate Random Color for this line
		int r = rand() % 256;
		int g = rand() % 256;
		int b = rand() % 256;
		
		int x1 = 0;
		int y1 = (int)floor(line.rho / cos(line.ang*M_PI / 180) + 0.5);
		int x2 = imgSize.first - 1;
		int y2 = (int)floor((line.rho - x2*sin(line.ang*M_PI / 180)) / cos(line.ang*M_PI / 180) + 0.5);

		if (line.ang == 90) {
			int x = (int)(line.rho + 0.5);
			for (int y = 0; y < imgSize.second; y++) {
				result[x][y].setColor(r, g, b);
			}
		}
		else if (x1 == x2) {
			if (y1 > y2) std::swap(y1, y2);
			for (int y = y1; y <= y2; y++) {
				result[x1][y].setColor(r, g, b);
			}
		}
		else {
			double m = (double)(y2 - y1) / (x2 - x1);
			if (m > -1 && m < 1) {
				if (x1 > x2) {
					std::swap(x1, x2);
					std::swap(y1, y2);
				}
				for (int x = x1; x <= x2; x++) {
					int y = (int)floor(m*(x - x1) + y1 + 0.5);
					if (y >= 0 && y < imgSize.second) {
						result[x][y].setColor(r, g, b);
					}
				}
			}
			else {
				if (y1 > y2) {
					std::swap(x1, x2);
					std::swap(y1, y2);
				}
				for (int y = y1; y <= y2; y++) {
					int x = (int)floor((y - y1) / m + x1 + 0.5);
					if (y >= 0 && y < imgSize.second) {
						result[x][y].setColor(r, g, b);
					}
				}
			}

		}
	}
	
	return result;
}

imgctrl::Image::Image(const std::vector<std::vector<Color>> &image)
{
	m_image = image;
}

imgctrl::Image::~Image()
{
}


imgctrl::Image imgctrl::Image::load(const std::string& filename)
{
	cv::Mat cvImage = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	const cv::Size size = cvImage.size();
	std::vector< std::vector<Color> > data(size.width, std::vector<Color>(size.height));

	for (unsigned int i = 0; i < size.width; i++) {
		for (unsigned int j = 0; j < size.height; j++) {
			cv::Vec3b& currentColor = cvImage.at<cv::Vec3b>(cv::Point(i, j));
			data[i][j].setRed(currentColor[Color::kRedIdx]);
			data[i][j].setGreen(currentColor[Color::kGreenIdx]);
			data[i][j].setBlue(currentColor[Color::kBlueIdx]);
		}
	}

	return Image(data);
}

imgctrl::Image::Image(const std::pair<size_t, size_t>& size)
{
	m_image.assign(size.first, std::vector<Color>(size.second));
}

void imgctrl::Image::save(const std::string& filename) const
{
	// Not Implement
}

size_t imgctrl::Image::getHeight() const
{
	if (m_image.size() == 0) return 0;
	return m_image[0].size();
}

size_t imgctrl::Image::getWidth() const
{
	return m_image.size();
}

std::vector<imgctrl::Color>& imgctrl::Image::operator[](const unsigned int& idx)
{
	return m_image[idx];
}

const std::vector<imgctrl::Color>& imgctrl::Image::operator[](const unsigned int& idx) const
{
	return m_image[idx];
}

std::pair<size_t, size_t> imgctrl::Image::getSize() const
{
	return std::pair<size_t, size_t>({getWidth(), getHeight()});
}

imgctrl::Image::operator cv::Mat() const
{
	const std::pair<size_t, size_t> size = getSize();
	cv::Size resultSize(size.first, size.second);
	cv::Mat result(resultSize, CV_8UC3);

	for (unsigned int i = 0; i < size.first; i++) {
		for (unsigned int j = 0; j < size.second; j++) {
			cv::Vec3b& currentColor = result.at<cv::Vec3b>(cv::Point(i, j));
			currentColor[2] = m_image[i][j].getRed();
			currentColor[1] = m_image[i][j].getGreen();
			currentColor[0] = m_image[i][j].getBlue();
		}
	}
	return result;
}

imgctrl::Color::Color() {
	setColor(0, 0, 0);
}

imgctrl::Color::Color(const COLORDATA& r, const COLORDATA& g, const COLORDATA& b)
{
	setRed(r);
	setGreen(g);
	setBlue(b);
}

imgctrl::Color::~Color()
{
}

void imgctrl::Color::setRed(const COLORDATA& r)
{
	m_color[kRedIdx] = r;
}

void imgctrl::Color::setGreen(const COLORDATA& g)
{
	m_color[kGreenIdx] = g;
}

void imgctrl::Color::setBlue(const COLORDATA& b)
{
	m_color[kBlueIdx] = b;
}

void imgctrl::Color::setColor(const COLORDATA & r, const COLORDATA & g, const COLORDATA & b)
{
	setRed(r);
	setGreen(g);
	setBlue(b);
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

imgctrl::LineParam::LineParam()
{
}

imgctrl::LineParam::LineParam(std::initializer_list<double> data)
{
	assert(data.size() == 2);
	const double* ptr = data.begin();
	this->rho = ptr[0];
	this->ang = ptr[1];
}


imgctrl::LineParam::~LineParam()
{
}

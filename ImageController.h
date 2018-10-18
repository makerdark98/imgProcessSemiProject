#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <algorithm>
#include <atlimage.h>
#include "opencv2/highgui/highgui.hpp"

namespace imgctrl {
	using COLORDATA = BYTE[3];

	class Color {
	private:
		COLORDATA m_color;

		static const uint8_t kRedIdx = 2;
		static const uint8_t kGreenIdx = 1;
		static const uint8_t kBlueIdx = 0;

	public:
		Color();
		Color(const BYTE& r, const BYTE& g, const BYTE& b);
		~Color();

		void setRed(BYTE r);
		void setGreen(BYTE g);
		void setBlue(BYTE b);

		BYTE getRed() const;
		BYTE getGreen() const;
		BYTE getBlue() const;
	};

	class Image {
	private:
		std::vector< std::vector<Color> > m_image;
		Image(const std::vector< std::vector<Color> >& image);

	public:
		static Image load(const std::string& filename);
		Image(std::pair<size_t, size_t> size);
		~Image();
		void save(const std::string& filename) const;

		std::pair<size_t, size_t> getSize() const;
		size_t getHeight() const;
		size_t getWidth() const;

		std::vector<Color>& operator[](const unsigned int& idx);
		const std::vector<Color>& operator[](const unsigned int& idx) const;
		operator cv::Mat() const;
	};

	class LineParam {
	public:
		double rho;
		double ang;
		LineParam();
		LineParam(std::initializer_list<double> data);
		~LineParam();
	};


	class ImageController
	{
	private:
		int threshold = 128;
		BYTE binaryBlack = 0;
		BYTE binaryWhite = 255;

	public:
		ImageController();
		~ImageController();

		BYTE getThreshold() const;
		void setThreshold(const int& threshold);

		Image getBinarization(const Image& original) const;
		Image getBlur(const Image& original) const;
		Image getSharpening(const Image& original) const;
		Image getGrayScale(const Image& original) const;
		Image getConvolution(const Image& original, const std::vector<std::vector<double> > &filter) const;
		Image getMarkedImage(const Image& original, const std::vector<std::pair<unsigned int, unsigned int> > &markPositions, const unsigned int& markSize = 5);

		std::vector< std::pair<unsigned int,unsigned int> > getHarrisCorner(const Image& image) const;
		std::vector< imgctrl::LineParam> getHoughLine(const Image& image) const;
	};

}
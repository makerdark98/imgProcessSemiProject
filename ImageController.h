#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <algorithm>
#include <atlimage.h>
#include "opencv2/highgui/highgui.hpp"

namespace imgctrl {
	
	using COLORDATA = BYTE;
	using COLORBGR = COLORDATA[3];

	class Color {
	private:
		COLORBGR m_color;

	public:
		static const uint8_t kRedIdx = 2;
		static const uint8_t kGreenIdx = 1;
		static const uint8_t kBlueIdx = 0;

		Color();
		Color(const COLORDATA& r, const COLORDATA& g, const COLORDATA& b);
		~Color();

		void setRed(const COLORDATA& r);
		void setGreen(const COLORDATA& g);
		void setBlue(const COLORDATA& b);
		void setColor(const COLORDATA& r, const COLORDATA& g, const COLORDATA& b);

		COLORDATA getRed() const;
		COLORDATA getGreen() const;
		COLORDATA getBlue() const;
	};

	/* ==== !! Must Use load() to generate image */
	class Image {
	private:
		std::vector< std::vector<Color> > m_image;
		Image(const std::vector< std::vector<Color> >& image); 

	public:
		static Image load(const std::string& filename); 
		Image(const std::pair<size_t, size_t>& size);
		~Image();
		void save(const std::string& filename) const; // Not Implement

		/* A return value's first is width and second is height */
		std::pair<size_t, size_t> getSize() const;
		size_t getHeight() const;
		size_t getWidth() const;

		std::vector<Color>& operator[](const unsigned int& idx);
		const std::vector<Color>& operator[](const unsigned int& idx) const;
		operator cv::Mat() const;
	};

	/* For HoughLine method */
	class LineParam {
	public:
		double rho;
		double ang;

		LineParam();
		LineParam(std::initializer_list<double> data);
		~LineParam();
	};


	// I don't care optimization and speed, So all control method is return data, not edit original data
	class ImageController
	{
	private:
		int m_threshold = 128;	// 128 is for binarization, if you use others, must set threshhold (e.g 20000)
		static const COLORDATA kBlackBinary = 0;
		static const COLORDATA kWhiteBinary = 255;

	public:
		ImageController();
		~ImageController();

		COLORDATA getThreshold() const;
		void setThreshold(const int& threshold);

		Image getBinarization(const Image& original) const;
		Image getBlur(const Image& original) const;
		Image getSharpening(const Image& original) const;
		Image getGrayScale(const Image& original) const;
		Image getConvolution(const Image& original, const std::vector<std::vector<double> > &filter) const;
		Image getMarkedImage(const Image& original, const std::vector<std::pair<unsigned int, unsigned int> > &markPositions, const unsigned int& markSize = 5);
		Image getCompositiion(const Image& firstImage, const Image& secondImage) const;
		Image getLinedImage(const Image& original, const std::vector<imgctrl::LineParam>& lines) const;

		std::vector< std::pair<unsigned int,unsigned int> > getHarrisCorner(const Image& image) const;
		std::vector< imgctrl::LineParam> getHoughLine(const Image& image) const;
	};

}
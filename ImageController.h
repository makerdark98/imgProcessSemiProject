#pragma once
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include <algorithm>
#include <atlimage.h>

namespace imgctrl {
	using COLORDATA = BYTE[3];

	class Color {
	private:
		COLORDATA m_color;

		static const uint8_t kRedIdx = 2;
		static const uint8_t kGreenIdx = 1;
		static const uint8_t kBlueIdx = 0;

	public:
		friend class ImageController;
		Color();
		Color(BYTE r, BYTE g, BYTE b);
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
		Image(std::vector< std::vector<Color> > image);

	public:
		static Image load(std::string filename);
		friend class ImageController;
		~Image();
		void save(std::string filename) const;

		std::pair<size_t, size_t> getSize() const;
		size_t getHeight() const;
		size_t getWidth() const;

		operator cv::Mat() const;
	};

	class ImageController
	{
	private:
		BYTE threshold = 128;
		BYTE binaryBlack = 0;
		BYTE binaryWhite = 255;
		
	public:
		ImageController();
		~ImageController();

		BYTE getThreshold() const;
		void setThreshold(const BYTE& threshold);

		Image getBinarization(const Image& original) const;
		Image getGrayScale(const Image& original) const;

		std::vector< std::pair<unsigned int,unsigned int> > getHarrisCorner(const Image& image) const;

		Image getMarkedImage(const Image& original, std::vector<std::pair<unsigned int, unsigned int> > &markPositions, unsigned int markSize = 5);

	};

}
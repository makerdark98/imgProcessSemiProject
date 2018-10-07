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
		Color();
		Color(BYTE r, BYTE g, BYTE b);
		~Color();

		void setRed(BYTE r);
		void setGreen(BYTE g);
		void setBlue(BYTE b);

		BYTE getRed() const;
		BYTE getGreen() const;
		BYTE getBlue() const;

		friend class ImageController;

	};

	class Image {
	private:
		std::vector< std::vector<Color> > m_image;
		Image(std::vector< std::vector<Color> > image);

	public:
		~Image();
		static Image load(std::string filename);
		void save(std::string filename) const;

		size_t getHeight() const;
		size_t getWidth() const;
		std::pair<size_t, size_t> getSize() const;
		friend class ImageController;

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
		std::vector< std::pair<int, int> > getHarrisCorner(const Image& image) const;

	};

}
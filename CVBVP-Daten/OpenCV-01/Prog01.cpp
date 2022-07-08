
// OpenCV_Helloworld 

#include <opencv2/opencv.hpp>

int main() {
	cv::Mat image;
	image = cv::imread("Daten/Img01a.jpg");
	if (!image.data) {
		printf("Error: Couldn't open the image file.\n");
		return 1;
	}
	cv::namedWindow("Image:");
	cv::imshow("Image:", image); 
	cv::waitKey(0);
	cv::destroyWindow("Image:");
	return 0;
}

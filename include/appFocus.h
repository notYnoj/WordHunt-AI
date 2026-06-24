#pragma once
#include <opencv2/opencv.hpp>

void bringAppToForeground(const char* appName);
cv::Mat captureRegion(int x, int y, int width, int height);
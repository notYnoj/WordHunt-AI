#pragma once
#include <iostream>
#include <CoreGraphics/CoreGraphics.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>

void listAllWindows();
std::pair<std::pair<int, int>, std::pair<int, int>> getAppCoordinates(const std::string& app);
cv::Mat getSnapShot(const std::pair<int, int>& x1y1, const std::pair<int, int>& x2y2);
cv::Mat grayScale(cv::Mat image);
cv::Mat findGrid(cv::Mat image);
std::pair<std::vector<cv::Mat>, std::vector<cv::Rect>> extractLetters(cv::Mat processedGrid, cv::Mat image, double minPercent, double maxPercent);
std::pair<std::vector<cv::Mat>, std::vector<std::pair<int, int>>>  wrapperGetLetters(std::string s);
void saveLetters(const std::vector<cv::Mat>& letters, unsigned int& t, const std::string& outDir = "../../images");
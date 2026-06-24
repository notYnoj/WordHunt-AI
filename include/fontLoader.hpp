#pragma once
#include "cnn.hpp"
#include "loader.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

inline cv::Mat augmentLetter(const cv::Mat& src, std::mt19937& rng){
    std::uniform_real_distribution<float> angleDist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> shiftDist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> scaleDist(0.95f, 1.05f);
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, angleDist(rng), scaleDist(rng));
    rot.at<double>(0, 2) += shiftDist(rng);
    rot.at<double>(1, 2) += shiftDist(rng);
    cv::Mat warped;
    cv::warpAffine(src, warped, rot, src.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::Mat out;
    cv::threshold(warped, out, 127, 255, cv::THRESH_BINARY);
    return out;
}



inline void countFontImages(const std::string& root){
    std::cout << "Letter counts in " << root << ":\n";
    size_t total = 0;
    for(char c = 'A'; c <= 'Z'; c++){
        std::string dir = root + "/" + std::string(1, c);
        size_t count = 0;
        if(fs::exists(dir)){
            for(const auto& entry : fs::directory_iterator(dir)){
                if(entry.path().extension() == ".png") count++;
            }
        }
        total += count;
        std::cout << c << ": " << count << "\n";
    }
    std::cout << "Total: " << total << "\n";
}


template <typename T>
typename CNN<T>::Dataset loadBalancedDataset(const std::string& fontRoot, const std::string& emnistImagePath,
                                             const std::string& emnistLabelPath, int targetPerLetter = 15,
                                             int augmentPerImage = 25)
{
    typename CNN<T>::Dataset dataset;
    std::mt19937 rng(std::random_device{}());

    auto pushSample = [&](const cv::Mat& img, int classIdx){
        std::vector<T> data(28 * 28);
        for(int r = 0; r < 28; r++)
            for(int c = 0; c < 28; c++)
                data[r * 28 + c] = static_cast<T>(img.at<uchar>(r, c)) / static_cast<T>(255);
        Tensor<T> tensor(std::vector<long long>{28, 28}, data);
        dataset.push_back({ {tensor}, classIdx });
    };

    std::array<std::vector<cv::Mat>, 26> fontImages;
    for(char c = 'A'; c <= 'Z'; c++){
        std::string dir = fontRoot + "/" + std::string(1, c);
        if(!fs::exists(dir)) continue;
        int classIdx = c - 'A';
        for(const auto& entry : fs::directory_iterator(dir)){
            if(entry.path().extension() != ".png") continue;
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
            if(img.empty() || img.rows != 28 || img.cols != 28){
                std::cerr << "skipping bad image because either empty or not 28x28: " << entry.path() << "\n";
                continue;
            }
            fontImages[classIdx].push_back(img);
        }
        std::shuffle(fontImages[classIdx].begin(), fontImages[classIdx].end(), rng);
    }

    std::array<int, 26> needed{};
    int classesStillNeeded = 0;
    for(int i = 0; i < 26; i++){
        needed[i] = std::max(0, targetPerLetter - (int)fontImages[i].size());
        if(needed[i] > 0) classesStillNeeded++;
    }

    std::array<std::vector<cv::Mat>, 26> emnistImages;
    if(classesStillNeeded > 0){
        std::cout << "Pulling EMNIST rehearsal samples for " << classesStillNeeded << " short letters...\n";
        std::ifstream imgFile(emnistImagePath, std::ios::binary);
        std::ifstream lblFile(emnistLabelPath, std::ios::binary);
        if(!imgFile || !lblFile) throw std::runtime_error("Cannot open EMNIST files");

        uint32_t imgMagic = readBigEndian(imgFile);
        uint32_t numImages = readBigEndian(imgFile);
        uint32_t rows = readBigEndian(imgFile);
        uint32_t cols = readBigEndian(imgFile);
        uint32_t lblMagic = readBigEndian(lblFile);
        uint32_t numLabels = readBigEndian(lblFile);
        if(imgMagic != 2051 || lblMagic != 2049 || numImages != numLabels)
            throw std::runtime_error("Bad EMNIST file headers");

        std::vector<uint8_t> pixelBuf(rows * cols);
        for(uint32_t n = 0; n < numImages && classesStillNeeded > 0; n++){
            imgFile.read(reinterpret_cast<char*>(pixelBuf.data()), rows * cols);
            uint8_t label;
            lblFile.read(reinterpret_cast<char*>(&label), 1);
            int classIdx = static_cast<int>(label) - 1;  // 1-26 => 0-25
            if(classIdx < 0 || classIdx >= 26) continue;
            if((int)emnistImages[classIdx].size() >= needed[classIdx]) continue;

            cv::Mat img(rows, cols, CV_8UC1);
            for(uint32_t r = 0; r < rows; r++)
                for(uint32_t c = 0; c < cols; c++)
                    img.at<uchar>(r, c) = pixelBuf[c * rows + r];

            emnistImages[classIdx].push_back(img);
            if((int)emnistImages[classIdx].size() == needed[classIdx]) classesStillNeeded--;
        }
    }

    for(int classIdx = 0; classIdx < 26; classIdx++){
        char letter = 'A' + classIdx;
        int useCount = std::min((int)fontImages[classIdx].size(), targetPerLetter);

        for(int i = 0; i < useCount; i++){
            pushSample(fontImages[classIdx][i], classIdx);
            for(int k = 0; k < augmentPerImage; k++)
                pushSample(augmentLetter(fontImages[classIdx][i], rng), classIdx);
        }

        int filled = (int)emnistImages[classIdx].size();
        for(int i = 0; i < filled; i++)
            pushSample(emnistImages[classIdx][i], classIdx);

        std::cout << letter << ": " << useCount << " font + " << filled << " emnist"
                  << (useCount + filled < targetPerLetter ? "  (still short!)" : "") << "\n";
    }
    std::cout << "Total dataset size: " << dataset.size() << "\n";
    return dataset;
}

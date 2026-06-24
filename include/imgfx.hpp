#pragma once
#include "signal.hpp"
#include <cassert>
#include <opencv2/opencv.hpp>

template <typename T>
inline std::vector<Tensor<T>> matToTensor(const cv::Mat& img) {
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);

    std::vector<Tensor<T>> result;
    std::vector<long long> shape = {img.rows, img.cols};

    for (int c = 0; c < 3; c++) {
        std::vector<T> data;
        data.reserve(img.rows * img.cols);
        for (int i = 0; i < img.rows; i++)
            for (int j = 0; j < img.cols; j++)
                data.push_back(static_cast<T>(channels[c].at<uchar>(i, j)) / static_cast<T>(255));
        result.push_back(Tensor<T>(shape, data));
    }
    return result;
}

template <typename T>
inline cv::Mat tensorToMat(const std::vector<Tensor<T>>& channels) {
    std::vector<cv::Mat> mats;
    for (const auto& channel : channels) {
        const auto& shape = channel.getShape();
        const auto& data = channel.getData();
        assert(shape.size() == 2); //each channel should only have a HxW Tensor
        cv::Mat mat(shape[0], shape[1], CV_8UC1);
        for (int i = 0; i < shape[0]; i++)
            for (int j = 0; j < shape[1]; j++)
                mat.at<uchar>(i, j) = static_cast<uchar>(
                    std::clamp(data[i * shape[1] + j], static_cast<T>(0), static_cast<T>(1)) * static_cast<T>(255)
                );
        mats.push_back(mat);
    }
    cv::Mat ret;
    cv::merge(mats, ret);
    return ret;
}

//kernel is the blur level
template <typename T>
inline Tensor<T> blur(const Tensor<T>& input, size_t kernel) {
    const auto& inputShape = input.getShape();
    std::vector<long long> kernelShape;
    size_t mx = 1;
    for (const long long& i: inputShape) {
        if (i < static_cast<long long>(kernel)) {
            throw std::out_of_range("Kernel is bigger than input shape");
        }
        kernelShape.push_back(static_cast<long long>(kernel));
        mx*=kernel;
    }
    std::vector<T> kernelData(mx, static_cast<T>(1) / static_cast<T>(mx));

    Tensor<T> k(kernelShape, kernelData);

    return crossCorrelate(input, k, Mode::Same);
}
template<typename T>
inline cv::Mat blur(const cv::Mat& input, size_t kernelSize) {
    auto channels = matToTensor<T>(input);
    std::vector<Tensor<T>> blurred;
    for (const auto& channel : channels)
        blurred.push_back(blur<T>(channel, kernelSize));
    return tensorToMat<T>(blurred);
}

inline cv::Mat toGray(const cv::Mat& img) {
    if (img.channels() == 1) return img.clone();
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

inline cv::Mat resizeSquare(const cv::Mat& img, int size = 28) {
    cv::Mat resized;
    int interp = (img.rows > size || img.cols > size) ? cv::INTER_AREA : cv::INTER_CUBIC;
    cv::resize(img, resized, cv::Size(size, size), 0, 0, interp);
    return resized;
}

template <typename T>
inline std::vector<Tensor<T>> matToGrayTensor(const cv::Mat& img) {
    std::vector<T> ret;
    ret.reserve(img.rows * img.cols);
    for (int i = 0; i < img.rows; i++) {
        const auto* row = img.ptr<uchar>(i);
        for (int j = 0; j < img.cols; j++)
            ret.push_back(static_cast<T>(row[j]) / static_cast<T>(255));
    }
    return {Tensor<T>({(long long)img.rows, (long long)img.cols}, ret)};
}

inline cv::Mat binarizeLetter(const cv::Mat& grayImage){
    cv::Mat binary;
    cv::threshold(grayImage, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    int whiteCount = cv::countNonZero(binary);
    int totalCount = binary.rows * binary.cols;
    if (whiteCount > totalCount / 2) {
        cv::bitwise_not(binary, binary);
    }
    return binary;
}

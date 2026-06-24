#pragma once
#include "fft.hpp"

enum class Mode {
    Full, //Everything
    Same, //Center is On
    Valid
};
inline size_t nextPow2(const size_t& n) {
    size_t p = 1;
    while (p < n) p<<=1;
    return p;
}

template <typename T>
inline Tensor<T> zeroPad(const Tensor<T>& input, const std::vector<long long>& target) {
    Tensor<T> ret(target, Init::Zero);
    const auto& inputShape = input.getShape();
    const auto& inputStrides = input.getStrides();
    const auto& inputSize = static_cast<size_t>(input.getSize());
    for (size_t i{}; i < inputSize; ++i) {
        size_t remain = i;
        std::vector<size_t> index(inputShape.size());
        for (size_t d = 0; d < inputShape.size(); d++) {
            index[d] = remain/inputStrides[d];
            remain%=inputStrides[d];
        }
        ret.place(index, input.get(index));
    }
    return ret;
}

template<typename T>
inline Tensor<T> convolveFFT(const Tensor<T>& input, const Tensor<T>& kernel, Mode mode = Mode::Full) {
    const auto& shapeInput = input.getShape();
    const auto& shapeKernel = kernel.getShape();
    if (shapeInput.size() != shapeKernel.size()) {
        throw std::out_of_range("Input and Kernel sizes do not match");
    }
    for (size_t i{}; i<shapeKernel.size(); i++) {
        if (shapeInput[i] < shapeKernel[i]) {
            throw std::out_of_range("Kernel has a bigger size than Input");
        }
    }

    std::vector<long long> paddedShape(shapeInput.size());
    for (size_t i{}; i<shapeInput.size(); i++) {
        paddedShape[i] = nextPow2(shapeInput[i] + shapeKernel[i] - 1);
    }

    Tensor<T> inputPadded = zeroPad(input, paddedShape);
    Tensor<T> kernelPadded = zeroPad(kernel, paddedShape);

    Tensor<Complex> inputFFT = fftND(inputPadded);
    Tensor<Complex> kernelFFT = fftND(kernelPadded);

    Tensor<Complex> product = inputFFT * kernelFFT;

    Tensor<Complex> result = ifftND(product);

    const auto& resultData = result.getData();
    std::vector<T> realData(resultData.size());
    for (size_t i{}; i < resultData.size(); i++) {
        realData[i] = static_cast<T>(resultData[i].real());
    }
    Tensor<T> realResult(paddedShape, realData);

    // slice to mode
    std::vector<size_t> start(shapeInput.size());
    std::vector<size_t> sizes(shapeInput.size());
    for (size_t d = 0; d < shapeInput.size(); d++) {
        if (mode == Mode::Full) {
            start[d] = 0;
            sizes[d] = shapeInput[d] + shapeKernel[d] - 1;
        } else if (mode == Mode::Same) {
            start[d] = (shapeKernel[d] - 1) / 2;
            sizes[d] = shapeInput[d];
        } else {
            start[d] = shapeKernel[d] - 1;
            sizes[d] = shapeInput[d] - shapeKernel[d] + 1;
        }
    }
    return realResult.slice(start, sizes);
}

template <typename T>
inline Tensor<T> crossCorrelateFFT(const Tensor<T>& input, const Tensor<T>& kernel, Mode mode = Mode::Same) {
    return (convolve(input, kernel.flip(), mode));
}

template <typename T>
inline Tensor<T> crossCorrelate(const Tensor<T>& input, const Tensor<T>& kernel, Mode mode = Mode::Same) {
    const auto& inShape = input.getShape();
    const auto& kShape  = kernel.getShape();
    if (inShape.size() != 2 || kShape.size() != 2)
        throw std::out_of_range("direct crossCorrelate is 2D only");

    long long H = inShape[0], W = inShape[1];
    long long KH = kShape[0], KW = kShape[1];

    const auto& in = input.getData();
    const auto& k  = kernel.getData();
    long long fullH = H + KH - 1;
    long long fullW = W + KW - 1;
    long long outH, outW, startI, startJ;
    if (mode == Mode::Full) {
        outH = fullH;  outW = fullW;  startI = 0;            startJ = 0;
    } else if (mode == Mode::Same) {
        outH = H;      outW = W;      startI = (KH - 1) / 2; startJ = (KW - 1) / 2;
    } else { // Valid
        outH = H - KH + 1; outW = W - KW + 1; startI = KH - 1; startJ = KW - 1;
    }

    std::vector<T> out(static_cast<size_t>(outH * outW), T{});

    for (long long oi = 0; oi < outH; oi++) {
        for (long long oj = 0; oj < outW; oj++) {
            long long fi = oi + startI;
            long long fj = oj + startJ;
            T sum = T{};
            for (long long ki = 0; ki < KH; ki++) {
                long long ii = fi - (KH - 1) + ki;
                if (ii < 0 || ii >= H) continue;
                for (long long kj = 0; kj < KW; kj++) {
                    long long jj = fj - (KW - 1) + kj;
                    if (jj < 0 || jj >= W) continue;
                    sum += in[ii * W + jj] * k[ki * KW + kj];
                }
            }
            out[oi * outW + oj] = sum;
        }
    }
    return Tensor<T>({outH, outW}, out);
}

template <typename T>
inline Tensor<T> convolve(const Tensor<T>& input, const Tensor<T>& kernel, Mode mode = Mode::Full) {
    return crossCorrelate(input, kernel.flip(), mode);
}
#pragma once
#include <vector>
#include <complex>
#include "tensor.hpp"

using Complex = std::complex<double>;
const double PI = std::acos(-1.0);
//Pad, radixFFT
inline bool isPowTwo(const size_t size) {
    return size > 0 && (size & (size - 1)) == 0;
}

inline void pad(std::vector<Complex>& input) {
    if (!isPowTwo(input.size())) {
        unsigned long long newSize = 1ULL;
        while (newSize < input.size()) {
            newSize <<= 1;
        }
        input.resize(newSize);
    }
}

inline std::vector<Complex> radixFFTHelper(std::vector<Complex>& F) {
    if (F.size() == 1) {
        return F;
    }
    const size_t n = F.size();
    std::vector<Complex> ret;
    ret.resize(F.size());
    //find even and find odd first
    std::vector<Complex> even;
    std::vector<Complex> odd;
    for (size_t i{}; i<F.size(); ++i) {
        if (i&1) {
            odd.push_back(F[i]);
        }else {
            even.push_back(F[i]);
        }
    }
    std::vector<Complex> Feven = radixFFTHelper(even);
    std::vector<Complex> Fodd = radixFFTHelper(odd);
    for (size_t i{}; i<F.size()/2; ++i) {
        Complex twiddle = std::exp(Complex(0, -2 * PI * static_cast<double>(i) / static_cast<double>(n))); //e^ik/n;
        ret[i] = Feven[i] + twiddle * Fodd[i];
        ret[i + (n/2)] = Feven[i] - twiddle * Fodd[i];
    }
    return ret;
}

inline std::vector<Complex> radixFFT(std::vector<Complex>& F) {
    pad(F);
    std::vector<Complex> ret = radixFFTHelper(F);
    return ret;
}

inline std::vector<Complex> radixIFFT(std::vector<Complex>& F) {
    for (auto& c: F) {
        c = std::conj(c);
    }

    std::vector<Complex> ret = radixFFT(F);
    for (auto& c: ret) c = (std::conj(c) / static_cast<double>(F.size()));
    return ret;
}

template <Numeric T>
inline Tensor<Complex> fftND(const Tensor<T>& input) {
    const auto& shape = input.getShape();
    const auto& data = input.getData();
    const auto& strides = input.getStrides();

    std::vector<Complex> complexData(data.size());
    for (size_t i{}; i<data.size(); ++i) {
        complexData[i] = Complex(static_cast<double>(data[i]), 0.0);
    }
    Tensor<Complex> ret(shape, complexData);
    for (int dim = 0; dim < static_cast<int>(shape.size()); dim++) {
        for (size_t i{}; i < ret.getSize(); i++) {
            std::vector<size_t> multiIdx(shape.size());
            size_t remain = i;
            for (size_t d = 0; d < shape.size(); d++) {
                multiIdx[d] = remain / strides[d]; //strides should be equal
                remain %= strides[d];
            }
            if (multiIdx[dim] != 0) {
                continue;
            }
            std::vector<Complex> slice(shape[dim]);
            for (size_t j = 0; j < shape[dim]; j++) {
                multiIdx[dim] = j;
                slice[j] = ret.get(multiIdx);
            }
            std::vector<Complex> transformed = radixFFT(slice);
            for (size_t j = 0; j < shape[dim]; j++) {
                multiIdx[dim] = j;
                ret.place(multiIdx, transformed[j]);
            }
        }
    }
    return ret;
}

template <typename T>
inline Tensor<Complex> ifftND(const Tensor<T>& input) {
    const auto& shape = input.getShape();
    const auto& data = input.getData();
    const auto& strides = input.getStrides();
    std::vector<Complex> complexData;
    complexData.reserve(data.size());
    for (const auto& d : data) {
        if constexpr (std::is_same_v<T, Complex>) {
            complexData.push_back(d);
        } else {
            complexData.emplace_back(static_cast<double>(d), 0.0);
        }
    }
    Tensor<Complex> ret(shape, complexData);
    for (int dim{}; dim < static_cast<int>(shape.size()); dim++) {
        for (size_t node{}; node < ret.getSize(); node++) {
            std::vector<size_t> multiIdx(shape.size());
            size_t remain = node;
            for (size_t d = 0; d < shape.size(); d++) {
                multiIdx[d] = remain / strides[d];
                remain %= strides[d];
            }
            if (multiIdx[dim] != 0) {
                continue;
            }
            std::vector<Complex> slice(shape[dim]);
            for (size_t j{}; j < shape[dim]; j++) {
                multiIdx[dim] = j;
                slice[j] = ret.get(multiIdx);
            }
            std::vector<Complex> transformed = radixIFFT(slice);
            for (size_t j{}; j < shape[dim]; j++) {
                multiIdx[dim] = j;
                ret.place(multiIdx, transformed[j]);
            }
        }
    }
    return ret;
}
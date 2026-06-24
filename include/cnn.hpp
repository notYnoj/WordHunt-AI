#pragma once
#include "signal.hpp"
#include <cmath>
#include <numeric>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <thread>
#include <cstdint>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//cache for backprop
template <typename T>
struct ConvCache{
    std::vector<Tensor<T>> input;
    std::vector<Tensor<T>> preReLU;
};

template <typename T>
struct PoolCache{
    std::vector<Tensor<T>> input;
    std::vector<std::vector<size_t>> argmax;
};

template <typename T>
struct FCCache{
    std::vector<T> input;
    std::vector<T> preReLU;
    std::vector<std::vector<long long>> inShapes;
};


//base layer template
template <typename T>
struct BaseLayer{
    virtual std::vector<Tensor<T>> forward(const std::vector<Tensor<T>>& input) = 0;
    virtual std::vector<Tensor<T>> backward(const std::vector<Tensor<T>>& gradOut) = 0;
    virtual void stepEpoch() {}
    virtual void save(std::ofstream& f) const{}
    virtual void load(std::ifstream& f) {}
    virtual ~BaseLayer() = default;
};

template <typename T>
struct ConvLayer : BaseLayer<T>{
    size_t layersIn, layersOut, kernelSize;
    std::vector<Tensor<T>> filters;
    std::vector<T> biases;
    ConvCache<T> cache;
    T lr;
    T lrMin;
    T lrMax;
    size_t epoch;
    size_t totalEpochs;
    ConvLayer(size_t layersIn, size_t layersOut, size_t kernelSize, T lrMax = 1e-3, T lrMin = 1e-6, size_t totalEpochs = 100) :
    layersIn(layersIn),
    layersOut(layersOut),
    kernelSize(kernelSize),
    lr(lrMax),
    lrMin(lrMin),
    lrMax(lrMax),
    epoch(0),
    totalEpochs(totalEpochs){
        int fanIn = static_cast<int>(layersIn * kernelSize * kernelSize);
        for(size_t filter{}; filter < layersOut; filter++){
            filters.emplace_back(std::vector<long long>{static_cast<long long>(layersIn), static_cast<long long>(kernelSize), static_cast<long long>(kernelSize)}, Init::He, fanIn);
            biases.push_back(T{});
        }
    }
    void stepEpoch() override {
        epoch++;
        //cosine
        lr = lrMin + static_cast<T>(0.5) * (lrMax - lrMin) * (1 + std::cos(static_cast<T>(M_PI) * static_cast<T>(epoch) / static_cast<T>(totalEpochs)));
    }
    //One Thread should hand a section?
    //curFilter pos in write into is curFilter
    void forwardThread(const std::vector<Tensor<T>>& input, size_t curFilter, const Tensor<T>& filter, std::vector<Tensor<T>>& writeInto) {
        const std::vector<T>& filterData = filter.getData();
        long long area = kernelSize * kernelSize;
        for (size_t layer{}; layer < layersIn; layer++) {
            std::vector<T> kernelData(filterData.begin() + layer * area, filterData.begin() + (layer + 1) * area);
            Tensor<T> kernel({static_cast<long long>(kernelSize), static_cast<long long>(kernelSize)}, kernelData);
            writeInto[curFilter] += (crossCorrelate(input[layer], kernel, Mode::Valid));
        }

        auto& finalData = writeInto[curFilter].getData();
        for(T& i: finalData) {
            i += biases[curFilter];
        }
        cache.preReLU[curFilter] = writeInto[curFilter];
        for(T& i: finalData) {
            i = std::max(i, T{});
        }
    }
    std::vector<Tensor<T>> forward(const std::vector<Tensor<T>>& input) override{
        if(input[0].getShape().size() != 2){
            throw std::out_of_range("Input should be flat Feature Maps");
        }
        cache.input = input;

        long long H = input[0].getShape()[0];
        long long W = input[0].getShape()[1];
        long long outputH = H - kernelSize + 1;
        long long outputW = W - kernelSize + 1;
        std::vector<Tensor<T>> ret(layersOut, Tensor<T>({outputH, outputW}, Init::Zero));
        cache.preReLU.clear();
        cache.preReLU.resize(layersOut, Tensor<T>({outputH, outputW}, Init::Zero));
        std::vector<std::thread> threads;
        for (size_t curFilter{}; curFilter < filters.size(); curFilter++){
            threads.emplace_back(&ConvLayer::forwardThread, this,
                              std::cref(input), curFilter, std::cref(filters[curFilter]), std::ref(ret));
        }
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        return ret;
    }
    //pass in this to all of them
    void backwardThreadRELUThread(const std::vector<Tensor<T>>& gradOut, std::vector<Tensor<T>>& gradReLU, size_t layer){
        const auto& goData = gradOut[layer].getData();
        const auto& preData = cache.preReLU[layer].getData();
        std::vector<T> grad(goData.size());
        for(size_t i{}; i<goData.size(); i++){
            grad[i] = preData[i] > T{} ? goData[i] : T{};
        }
        gradReLU[layer] = Tensor<T>(gradOut[layer].getShape(), grad);
    }
    //threads.emplace_back(&ConvLayer::backwardInputGradThread,
    //this, std::ref(gradIn), std::cref(filters), std::cref(gradReLU), layer);
    void backwardInputGradThread(std::vector<Tensor<T>>& gradIn, const std::vector<Tensor<T>>& gradRelu, size_t layer){
        const long long area = kernelSize * kernelSize;
        for (size_t o{}; o < layersOut; o++) {
            std::vector<T> kData(filters[o].getData().begin() + layer*area,
                                 filters[o].getData().begin() + (layer+1)*area);
            Tensor<T> kernel({static_cast<long long>(kernelSize),
                              static_cast<long long>(kernelSize)}, kData);
            //convolve we do opposite of cross correlate
            Tensor<T> contrib = convolve(gradRelu[o], kernel, Mode::Full);
            gradIn[layer] += contrib;
        }
    }
    void backwardWeightsandBiases(size_t layer , const std::vector<Tensor<T>>& gradReLU){
            const long long area = kernelSize * kernelSize;
            auto& filterData = filters[layer].getData();
            T biasGrad = T{};
            for (const auto& v : gradReLU[layer].getData()) biasGrad += v;
            biases[layer] -= lr * biasGrad;
            /*Look for dLoss/dKernel[a][b] = dLoss/dOutput[i][j] (have) * dOutput[i][j]/dKernel[a][b]
             * (this is just input becuz) output = sum(input[i+ki][j+kj] * kernel[ki][kj]) d/dx is just input
             * So we are only looking at the one where input is equal to i+a, j+b since all other ones cancels out due to derivative of constant being 0
             * Now we have that dLoss/dKernel[a][b] = sum(grad[i][j] * input[a+i][b+j]) or js cross corrrelate again
             * */
            for (size_t c{}; c < layersIn; c++) {
                Tensor<T> wGrad = crossCorrelate(cache.input[c], gradReLU[layer], Mode::Valid);
                const auto& wgData = wGrad.getData();
                for (size_t k{}; k < static_cast<size_t>(area); k++)
                {
                    filterData[c * area + k] -= lr * wgData[k];
                }

        }
    }

    std::vector<Tensor<T>> backward(const std::vector<Tensor<T>>& gradOut) override{
        long long area = kernelSize * kernelSize;
        std::vector<std::thread> threads;
        //Relu Grad
        std::vector<Tensor<T>> gradReLU(layersOut, Tensor<T>());
        for(size_t layer{}; layer < layersOut; layer++){
            threads.emplace_back(&ConvLayer::backwardThreadRELUThread,
                this, std::cref(gradOut), std::ref(gradReLU), layer);
        }
        for (auto& thread : threads) thread.join();


        //compute input gradients to pass back
        long long H = cache.input[0].getShape()[0];
        long long W = cache.input[0].getShape()[1];
        std::vector<Tensor<T>> gradIn(layersIn, Tensor<T>({H,W}, Init::Zero));
        threads.clear();
        for (size_t layer{}; layer < layersIn; layer++) {
            threads.emplace_back(&ConvLayer::backwardInputGradThread,
                this, std::ref(gradIn), std::cref(gradReLU), layer);
        }
        for (auto& thread : threads) thread.join();



        threads.clear();
        //we then goin update weights and biases
        for (size_t layer{}; layer < layersOut; layer++)
        {
            threads.emplace_back(&ConvLayer::backwardWeightsandBiases, this, layer, std::cref(gradReLU));
        }
        for (auto& thread : threads) thread.join();
        return gradIn;
    }

    void save(std::ofstream& f) const override {
        for (const Tensor<T>& layer : filters) {
            const std::vector<T>& dLayer = layer.getData();
            for (const T& weight : dLayer) {
                f.write(reinterpret_cast<const char*>(&weight), sizeof(T));
            }
        }
        for (const T& weight : biases) {
            f.write(reinterpret_cast<const char*>(&weight), sizeof(T));
        }
    }

    void load(std::ifstream& f) override {
        for (Tensor<T>& layer : filters) {
            std::vector<T>& dLayer = layer.getData();
            for (T& weight : dLayer) {
                f.read(reinterpret_cast<char*>(&weight), sizeof(T));
            }
        }
        for (T& weight : biases) {
            f.read(reinterpret_cast<char*>(&weight), sizeof(T));
        }
    }
};



template <typename T>
struct maxPoolLayer : BaseLayer<T> {
    size_t poolSize;
    PoolCache<T> cache;

    maxPoolLayer(size_t poolSize = 2) : poolSize(poolSize){}
    void forwardThread(const Tensor<T>& layer, size_t idx, std::vector<Tensor<T>>& ret){
        if (layer.getShape().size() != 2)
        {
            throw std::out_of_range("Input should be a flat feature map");
        }
        const long long H = layer.getShape()[0];
        const long long W = layer.getShape()[1];
        const long long retH = H/poolSize;
        const long long retW = W/poolSize;
        const auto& data = layer.getData();
        std::vector<T> outData(retH * retW);
        std::vector<size_t> localArgmax(retH * retW);

        for(long long i{}; i < retH; i++){
            for(long long j{}; j < retW; j++){
                const auto si = static_cast<size_t>(i);
                const auto sj = static_cast<size_t>(j);
                const auto sW = static_cast<size_t>(W);

                size_t bestIdx = (si*poolSize)*sW + (sj*poolSize);
                T mx = data[bestIdx];
                for(size_t dx{}; dx < poolSize; dx++){
                    for(size_t dy{}; dy < poolSize; dy++){
                        size_t idx2 = (si*poolSize + dx)*sW + (sj*poolSize + dy);
                        if(data[idx2] > mx){ mx = data[idx2]; bestIdx = idx2; }
                    }
                }
                outData[si*static_cast<size_t>(retW) + sj] = mx;
                localArgmax[si*static_cast<size_t>(retW) + sj] = bestIdx;
            }
        }
        cache.argmax[idx] = localArgmax;
        ret[idx] = Tensor<T>(std::vector<long long>{retH, retW}, outData);

    }
    std::vector<Tensor<T>> forward(const std::vector<Tensor<T>>& input) override{
        cache.input = input;
        cache.argmax.assign(input.size(), {});

        std::vector<Tensor<T>> ret(input.size(), Tensor<T>());
        std::vector<std::thread> threads;
        for (size_t layer{}; layer < input.size(); layer++)
        {
            //pass in layer, idx, ret
            threads.emplace_back(&maxPoolLayer::forwardThread, this, std::cref(input[layer]), layer, std::ref(ret));
        }
        for (auto& thread : threads) thread.join();

        return ret;
    }
    void backwardThread(size_t c, const std::vector<Tensor<T>>& gradOut, std::vector<Tensor<T>>& gradIn)
    {
        const auto& shape = cache.input[c].getShape();
        Tensor<T> grad(shape, Init::Zero);
        auto& gradData = grad.getData();
        const auto& goData = gradOut[c].getData();
        for (size_t i{}; i < goData.size(); i++)
            gradData[cache.argmax[c][i]] += goData[i];
        gradIn[c] = grad;
    }
    std::vector<Tensor<T>> backward(const std::vector<Tensor<T>>& gradOut) override {
        std::vector<Tensor<T>> gradIn(gradOut.size(), Tensor<T>());
        std::vector<std::thread> threads;
        for (size_t c{} ; c < gradOut.size(); c++)
        {
            threads.emplace_back(&maxPoolLayer::backwardThread, this, c, std::cref(gradOut), std::ref(gradIn));
        }
        for (auto& thread : threads) thread.join();
        return gradIn;
    }
};

template <typename T>
struct FCLayer: BaseLayer<T>{
    size_t inSize, outSize;
    bool relu;
    Tensor<T> weights, biases;
    FCCache<T> cache;
    T lr, lrMin, lrMax;
    size_t epoch, totalEpochs;

    FCLayer(size_t inSize, size_t outSize, bool relu = true, T lrMax = 1e-3, T lrMin = 1e-6, size_t totalEpochs = 100) :
    inSize(inSize),
    outSize(outSize),
    relu(relu),
    weights({static_cast<long long>(outSize), static_cast<long long>(inSize)}, Init::He, inSize),
    biases({static_cast<long long>(outSize)}, Init::Zero),
    lr(lrMax),
    lrMin(lrMin),
    lrMax(lrMax),
    epoch(0),
    totalEpochs(totalEpochs)
    {}

    void stepEpoch() override {
        epoch++;
        lr = lrMin + static_cast<T>(0.5) * (lrMax - lrMin) *
            (1 + std::cos(static_cast<T>(M_PI) * static_cast<T>(epoch) /  static_cast<T>(totalEpochs)));
    }
    void forwardChunk(const size_t startK, const size_t endK, const std::vector<T>& flat,
                  const std::vector<T>& filterData, const std::vector<T>& biasData,
                  std::vector<T>& retData, std::vector<T>& pre)
    {
        const size_t kernelArea = flat.size();
        for (size_t kernel = startK; kernel < endK; kernel++) {   // loop over this thread's range
            T total = biasData[kernel];
            const size_t base = kernel * kernelArea;
            for (size_t idx{}; idx < kernelArea; idx++)
                total += filterData[base + idx] * flat[idx];
            pre[kernel] = total;
            retData[kernel] = (relu ? std::max(total, T{}) : total);
        }
    }

    std::vector<Tensor<T>> forward(const std::vector<Tensor<T>>& input) override {
        std::vector<T> flat;
        cache.inShapes.clear();
        for (const auto& layer : input) {
            cache.inShapes.push_back(layer.getShape());
            for (const auto& data = layer.getData(); const T& ele : data)
                flat.push_back(ele);
        }
        if (flat.size() != inSize)
            throw std::out_of_range("InSize must be equal to layers * layerArea");
        cache.input = flat;

        std::vector<T> filterData = weights.getData();
        std::vector<T> biasData = biases.getData();
        std::vector<T> retData(outSize);
        std::vector<T> pre(outSize);

        const size_t nThreads = std::min(static_cast<size_t>(8), outSize);
        std::vector<std::thread> threads;
        const size_t chunk = (outSize + nThreads - 1) / nThreads;

        for (size_t t = 0; t < nThreads; t++) {
            size_t startK = t * chunk;
            size_t endK = std::min(startK + chunk, outSize);
            if (startK >= endK) break;
            threads.emplace_back(&FCLayer::forwardChunk, this, startK, endK,
                             std::cref(flat), std::cref(filterData), std::cref(biasData),
                             std::ref(retData), std::ref(pre));
        }
        for (auto& thread : threads) thread.join();

        cache.preReLU = pre;
        return {Tensor<T>({static_cast<long long>(outSize)}, retData)};
    }
    std::vector<Tensor<T>> backward(const std::vector<Tensor<T>>& gradOut) override {
        //weight Grad: dL/dW[i][j] = grad[i] * input[j]
        //input grad = dL/dInput[j] = sum(W[i][j] * grad[i]
        //bias grad: dL/dB[i] = grad[i]
        const std::vector<T>& goData = gradOut[0].getData();
        std::vector<T>& W = weights.getData();
        std::vector<T>& B = biases.getData();
        //RelU
        std::vector<T> grad(outSize);
        for (size_t idx{}; idx < outSize; idx++) {
            grad[idx] = (relu && cache.preReLU[idx] <= T{} ) ? 0 : goData[idx];
        }
        //inoput grad
        std::vector<T> gradIn(inSize, T{});
        for (size_t j{}; j < inSize; j++) {
            //for each filter (in pos filter * Size + j = this contrib)
            for (size_t i{}; i<outSize; i++) {
                gradIn[j] += (W[i*inSize+j] * grad[i]);
            }
        }
        //update weights + bias

        for (size_t i{}; i < outSize; i++) {
            B[i] -= lr * grad[i];
            for (size_t j{}; j < inSize; j++)
                W[i*inSize + j] -= lr * grad[i] * cache.input[j];
        }

        std::vector<Tensor<T>> ret;
        size_t offset = 0;
        for (const auto& shp : cache.inShapes) {
            size_t count = 1;
            for (long long d : shp) count *= static_cast<size_t>(d);
            std::vector<T> chunk(gradIn.begin() + offset, gradIn.begin() + offset + count);
            ret.emplace_back(shp, chunk);
            offset += count;
        }
        return ret;
    }
    /*Tensor<T> weights, biases;
    */
    void save(std::ofstream& f) const override {
        const std::vector<T>& dWeight = weights.getData();
        const std::vector<T>& dBiases = biases.getData();
        for (const T& weight :  dWeight) {
            f.write(reinterpret_cast<const char*>(&weight), sizeof(T));
        }
        for (const T& weight : dBiases) {
            f.write(reinterpret_cast<const char*>(&weight), sizeof(T));
        }
    }
    void load(std::ifstream& f) override{
        std::vector<T>& dWeight = weights.getData();
        std::vector<T>& dBiases = biases.getData();
        for (T& weight : dWeight) {
            f.read(reinterpret_cast<char*>(&weight), sizeof(T));
        }
        for (T& weight: dBiases) {
            f.read(reinterpret_cast<char*>(&weight), sizeof(T));
        }
    }
};

template <typename T>
struct CNN {
    //el calma pa
    std::vector<std::unique_ptr<BaseLayer<T>>> layers;

    //vector of Images -> Correct Letter
    using Dataset = std::vector<std::pair<std::vector<Tensor<T>>, int>>;
    void add(std::unique_ptr<BaseLayer<T>> layer) {
        //best practice is just to rob that bih here
        layers.push_back(std::move(layer));
    }
    Tensor<T> forward(const std::vector<Tensor<T>>& input) {
        std::vector<Tensor<T>> current = input;
        for (auto& layer : layers) {
            current = layer->forward(current);
        }
        std::vector<T> logits = current[0].getData();
        T maxVal = *std::max_element(logits.begin(), logits.end());
        T expSum = T{};
        for (T& v : logits) { v = std::exp(v - maxVal); expSum += v; }
        //probability spread
        for (T& v : logits) v /= expSum;
        return Tensor<T>(std::vector<long long>{static_cast<long long>(logits.size())}, logits);
    }


    void backward(const Tensor<T>& probs, int label) {
        std::vector<T> grad = probs.getData();
        grad[label] -= T{1};   //grad of label is correct is 1 then cross entropy
        std::vector<Tensor<T>> gradOut = { Tensor<T>({static_cast<long long>(grad.size())}, grad) };
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            gradOut = (*it)->backward(gradOut);
    }

    void stepEpoch() {
        //add 1 to epoch
        for (auto& layer : layers) layer->stepEpoch();
    }

    int predict(const std::vector<Tensor<T>>& input) {
        //With input try to predict with forward
        Tensor<T> probs = forward(input);
        const auto& d = probs.getData();
        return static_cast<int>(std::max_element(d.begin(), d.end()) - d.begin());
    }

    //forward-only pass over a dataset -> {avg loss, accuracy}
    std::pair<T, T> evaluate(Dataset& data) {
        T totalLoss = T{};
        int correct = 0;
        for (auto& [image, label] : data) {
            Tensor<T> probs = forward(image);
            const auto& p = probs.getData();
            totalLoss += -std::log(p[label] + T{1e-9});
            int pred = static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
            if (pred == label) correct++;
        }
        //proability of correct and loss average
        return { totalLoss / static_cast<T>( data.size() ), static_cast<T>( correct ) / static_cast<T>( data.size() ) };
    }

    void train(Dataset& trainData, size_t epochs, const std::string& path,  Dataset* valData = nullptr) {
        T vMnLoss, vMaxAcc, trMinLoss, trMaxAcc;
        std::cout<<'\n'<<"Input vMnLoss, vMaxAcc, trMinLoss, trMaxAcc or -1 if not wanted";

        std::cin>>vMnLoss;
        if (vMnLoss == -1)
        {
            vMnLoss = std::numeric_limits<T>::max();
            trMinLoss = std::numeric_limits<T>::max();
            vMaxAcc = std::numeric_limits<T>::lowest();
            trMaxAcc = std::numeric_limits<T>::lowest();
        }
        else
        {
            std::cin>>vMaxAcc>>trMinLoss>>trMaxAcc;
        }

        std::vector<size_t> idx(trainData.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng(std::random_device{}());

        for (size_t e = 0; e < epochs; e++) {
            //shuffle for some reason? idk literature told me it was best practice?
            std::ranges::shuffle(idx, rng);
            T totalLoss = T{};
            int correct = 0;
            for (size_t i : idx) {
                //for each image and label
                auto& [image, label] = trainData[i];
                Tensor<T> probs = forward(image);
                const auto& p = probs.getData();
                //will break if p = 0
                totalLoss += -std::log(p[label] + T{1e-9});
                int pred = static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
                if (pred == label) correct++;
                backward(probs, label);
            }
            stepEpoch();   // decay lr once per epoch


            if (valData) {
                T trLoss = totalLoss / trainData.size();
                T trAcc  = static_cast<T>(correct) / trainData.size();

                std::cout << "Epoch " << (e+1) << "/" << epochs
                          << " | loss " << trLoss
                          << " | acc " << (trAcc*100) << "%";
                auto [vLoss, vAcc] = evaluate(*valData);
                std::cout << " | val loss " << vLoss
                          << " | val acc " << (vAcc*100) << "%";
                if (vLoss < vMnLoss && vAcc > vMaxAcc && trLoss < trMinLoss && trAcc > trMaxAcc)
                {
                    save(path);
                    vMnLoss = vLoss;
                    vMaxAcc = vAcc;
                    trMinLoss = trLoss;
                    trMaxAcc = trAcc;
                    std::cout<<"Saved!"<<'\n';
                }
            }else
            {
                T trLoss = totalLoss / trainData.size();
                T trAcc  = static_cast<T>(correct) / trainData.size();

                std::cout << "Epoch " << (e+1) << "/" << epochs
                          << " | loss " << trLoss
                          << " | acc " << (trAcc*100) << "%";
                if (trLoss < trMinLoss && trAcc > trMaxAcc)
                {
                    std::cout<<"Saved!"<<'\n';
                    save(path);
                }
            }
            std::cout << "\n";
        }
    }

    //how well it does per lettter
    void perLetterAccuracy(Dataset& data, int numLetter = 26) {
        std::vector<int> total(numLetter, 0), correct(numLetter, 0);
        for (auto& [image, label] : data) {
            int pred = predict(image);
            ++total[label];
            if (pred == label) ++correct[label];
        }
        for (int c = 0; c < numLetter; c++) {
            const char letter = static_cast<char>('A' + c);
            const float acc = total[c] ? (100.0f * static_cast<float>(correct[c]) / static_cast<float>(total[c])) : 0.0f;
            std::cout << letter << ": " << acc << "%  (" << correct[c] << "/" << total[c] << ")\n";
        }
    }
    //just call the path where u have the weights in
    void save(const std::string& path) const{
        std::ofstream file(path+"weights.bin", std::ios::binary);
        if (!file.is_open()) {
            std::cerr<<"File did not open";
            return;
        }
        for (const std::unique_ptr<BaseLayer<T>>& layer: layers) {
            layer -> save(file);
        }
    }
    void load(const std::string& path = "../models/EMNIST/") {
        std::ifstream file(path+"weights.bin", std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open weights file at " + path + "weights.bin");
            return;
        }
        for (std::unique_ptr<BaseLayer<T>>& layer: layers) {
            layer -> load(file);
        }
        if (file.fail()) throw std::runtime_error("Weights file too small (macos v windows)? may not match?");
        file.peek();
        if (!file.eof()) std::cout << "Warning: leftover bytes architecture (macos v windows) may not match?\n";
        std::cout << "Loaded weights at " << path << "\n";
    }
};


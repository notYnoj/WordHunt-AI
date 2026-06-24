//#include "imgfx.hpp"
#include "cnn.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#define nl '\n'
#include "fontLoader.hpp"
#include "loader.hpp"
static float eps = 1e-2f;
/*void testSignal() {
    std::vector<long long> shape1 = {4};
    std::vector<float> d1 = {2,0,1,3};
    std::vector<float> d2 = {1,4,-2,2};
    Tensor<float> a(shape1, d1);
    Tensor<float> b(shape1, d2);

    auto full = convolve(a, b, Mode::Full);
    for (const auto& v : full.getData()) std::cout << std::round(v) << " ";
    std::cout << "\n";
    auto same = convolve(a, b, Mode::Same);
    for (const auto& v : same.getData()) std::cout << std::round(v) << " ";
    std::cout << "\n";
    auto valid = convolve(a, b, Mode::Valid);
    for (const auto& v : valid.getData()) std::cout << std::round(v) << " ";
    std::cout << "\n";

    std::vector<long long> shape2 = {3,3};
    std::vector<float> inputData = {1,2,3,4,5,6,7,8,9};
    std::vector<float> identity = {0,0,0,0,1,0,0,0,0};
    Tensor<float> img(shape2, inputData);
    Tensor<float> idk(shape2, identity);

    auto result = crossCorrelate(img, idk, Mode::Full);
    for (const auto& v : result.getData()) std::cout << std::round(v) << " ";
    std::cout << "\n";
}
void testImageFX() {
    std::cout << "Loading image..." << std::endl;
    cv::Mat img = cv::imread("../include/test.jpg");
    std::cout << "Loaded" << std::endl;
    if (img.empty()) {
        std::cout << "Failed to load image\n" << std::endl;
        return;
    }
    cv::Mat blurred = blur<float>(img, 15);
    cv::imshow("Original", img);
    cv::imshow("Blurred", blurred);
    cv::waitKey(0);
}*/
void testTensor() {
    std::vector<long long> shape = {2, 3};
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    Tensor<float> t(shape, data);
    float a = t[0][0];
    //indexing
    std::cout << typeid(a).name() << "\n";
    std::cout << a << "\n";
    std::cout << t[1][0] << "\n";
    std::cout << t[1][2] << "\n";

}
bool testFC() {
    FCLayer<float> fc(5, 3, false);
    Tensor<float> input({5}, {0.5f, -1.2f, 0.8f, 2.0f, -0.3f});
    std::vector<float> gd = {1.0f, -2.0f, 0.5f};

    auto loss = [&](const Tensor<float>& x){
        auto out = fc.forward({x});
        const auto& od = out[0].getData();
        float L=0; for(size_t i=0;i<od.size();i++) L += gd[i]*od[i];
        return L;
    };

    std::vector<float> W0 = fc.weights.getData();
    std::vector<float> B0 = fc.biases.getData();

    fc.forward({input});
    auto gi = fc.backward({Tensor<float>({3}, gd)});
    std::vector<float> aIn = gi[0].getData();
    std::vector<float> W1 = fc.weights.getData();
    std::vector<float> aW(W0.size());
    for(size_t i=0;i<W0.size();i++) aW[i] = (W0[i]-W1[i])/fc.lr;
    fc.weights.getData()=W0; fc.biases.getData()=B0;

    std::vector<float> nIn(input.getSize());
    for(size_t k=0;k<(size_t)input.getSize();k++){
        Tensor<float> xp=input, xm=input;
        xp.getData()[k]+=eps; xm.getData()[k]-=eps;
        nIn[k]=(loss(xp)-loss(xm))/(2*eps);
    }
    std::vector<float> nW(W0.size());
    for(size_t k=0;k<W0.size();k++){
        std::vector<float> Wp=W0,Wm=W0; Wp[k]+=eps; Wm[k]-=eps;
        fc.weights.getData()=Wp; float Lp=loss(input);
        fc.weights.getData()=Wm; float Lm=loss(input);
        nW[k]=(Lp-Lm)/(2*eps);
    }
    fc.weights.getData()=W0; fc.biases.getData()=B0;

    float di=0, dw=0;
    for(size_t i=0;i<nIn.size();i++) di=std::max(di,std::abs(nIn[i]-aIn[i]));
    for(size_t i=0;i<nW.size();i++) dw=std::max(dw,std::abs(nW[i]-aW[i]));
    std::cout<<"[fully connected layer]   input-grad diff "<<di<<"   weight-grad diff "<<dw<<"\n";
    return di<1e-2f && dw<1e-2f;
}

bool testConv() {
    ConvLayer<float> conv(2, 2, 3);
    for(size_t o=0;o<conv.filters.size();o++){
        auto& fd = conv.filters[o].getData();
        for(size_t k=0;k<fd.size();k++) fd[k] = 0.1f*((int)((o*3+k)%5)-2);
    }
    for(auto& b: conv.biases) b = 0.0f;

    std::vector<Tensor<float>> input;
    for(int c=0;c<2;c++){
        std::vector<float> d(25);
        for(int i=0;i<25;i++) d[i] = 0.1f*((i*(c+1))%7) - 0.3f;
        input.emplace_back(std::vector<long long>{5,5}, d);
    }
    std::vector<Tensor<float>> g;
    for(int o=0;o<2;o++){
        std::vector<float> d(9);
        for(int i=0;i<9;i++) d[i] = 0.1f*((i*(o+2))%5) - 0.2f;
        g.emplace_back(std::vector<long long>{3,3}, d);
    }
    auto& gref = g;

    auto loss = [&](const std::vector<Tensor<float>>& in){
        auto out = conv.forward(in);
        float L=0;
        for(size_t o=0;o<out.size();o++){
            const auto& od=out[o].getData(); const auto& gdv=gref[o].getData();
            for(size_t i=0;i<od.size();i++) L += gdv[i]*od[i];
        }
        return L;
    };

    std::vector<std::vector<float>> F0(conv.filters.size());
    for(size_t o=0;o<conv.filters.size();o++) F0[o]=conv.filters[o].getData();
    std::vector<float> B0 = conv.biases;


    auto out = conv.forward(input);

    auto gi = conv.backward(g);
    std::vector<std::vector<float>> aIn(gi.size());
    for(size_t c=0;c<gi.size();c++) aIn[c]=gi[c].getData();
    std::vector<std::vector<float>> aW(F0.size());
    for(size_t o=0;o<F0.size();o++){
        aW[o].resize(F0[o].size());
        auto& fn=conv.filters[o].getData();
        for(size_t k=0;k<F0[o].size();k++) aW[o][k]=(F0[o][k]-fn[k])/conv.lr;
    }
    for(size_t o=0;o<F0.size();o++) conv.filters[o].getData()=F0[o];
    conv.biases=B0;

    std::vector<std::vector<float>> nIn(input.size());
    for(size_t c=0;c<input.size();c++){
        nIn[c].resize(input[c].getSize());
        for(size_t k=0;k<(size_t)input[c].getSize();k++){
            auto ip=input, im=input;
            ip[c].getData()[k]+=eps; im[c].getData()[k]-=eps;
            nIn[c][k]=(loss(ip)-loss(im))/(2*eps);
        }
    }
    std::vector<std::vector<float>> nW(F0.size());
    for(size_t o=0;o<F0.size();o++){
        nW[o].resize(F0[o].size());
        for(size_t k=0;k<F0[o].size();k++){
            conv.filters[o].getData()=F0[o]; conv.filters[o].getData()[k]+=eps;
            float Lp=loss(input);
            conv.filters[o].getData()=F0[o]; conv.filters[o].getData()[k]-=eps;
            float Lm=loss(input);
            nW[o][k]=(Lp-Lm)/(2*eps);
            conv.filters[o].getData()=F0[o];
        }
    }

    float di=0, dw=0;
    for(size_t c=0;c<aIn.size();c++) for(size_t k=0;k<aIn[c].size();k++)
        di=std::max(di,std::abs(nIn[c][k]-aIn[c][k]));
    for(size_t o=0;o<aW.size();o++) for(size_t k=0;k<aW[o].size();k++)
        dw=std::max(dw,std::abs(nW[o][k]-aW[o][k]));
    std::cout<<"[conv] input-grad diff "<<di<<"   weight-grad diff "<<dw<<"\n";
    return di<2e-2f && dw<2e-2f;
}

bool testPool() {
    maxPoolLayer<float> pool(2);
    std::vector<float> d = {
        1,3,2,5, 4,8,7,6, 9,11,10,13, 12,16,15,14
    };
    std::vector<Tensor<float>> input;
    input.emplace_back(std::vector<long long>{4,4}, d);
    std::vector<float> gd = {1.0f,-2.0f,0.5f,3.0f};
    auto& gref = gd;

    auto loss=[&](const std::vector<Tensor<float>>& in){
        auto out=pool.forward(in);
        const auto& od=out[0].getData();
        float L=0; for(size_t i=0;i<od.size();i++) L+=gref[i]*od[i];
        return L;
    };

    pool.forward(input);
    auto gi=pool.backward({Tensor<float>({2,2}, gd)});
    std::vector<float> aIn=gi[0].getData();

    std::vector<float> nIn(input[0].getSize());
    for(size_t k=0;k<(size_t)input[0].getSize();k++){
        auto ip=input, im=input;
        ip[0].getData()[k]+=eps; im[0].getData()[k]-=eps;
        nIn[k]=(loss(ip)-loss(im))/(2*eps);
    }
    float di=0; for(size_t i=0;i<nIn.size();i++) di=std::max(di,std::abs(nIn[i]-aIn[i]));
    std::cout<<"[maxpool] input-grad diff "<<di<<"\n";
    return di<1e-2f;
}

void testFFT2() {
    std::vector<long long> xShape = {5, 5};
    std::vector<float> xData = {
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11,12,13,14,15,
            16,17,18,19,20,
            21,22,23,24,25
    };
    Tensor<float> X(xShape, xData);
    std::vector<long long> kShape = {3, 3};
    std::vector<float> kData = {
            0.2f, -0.1f, 0.3f,
            0.5f,  0.0f, -0.4f,
            -0.2f, 0.1f, 0.6f
    };
    Tensor<float> K(kShape, kData);
    std::vector<long long> gShape = {3, 3};
    std::vector<float> gData = {
            1.0f, -2.0f, 0.5f,
            0.3f,  1.5f, -1.0f,
            2.0f, -0.5f, 0.8f
    };
    Tensor<float> g(gShape, gData);
    // loss L = sum(g_i * Y_i), so dL/dY = g exactly
    auto computeLoss = [&](const Tensor<float>& input) {
        Tensor<float> Y = crossCorrelate(input, K, Mode::Valid);
        const auto& yd = Y.getData();
        const auto& gd = g.getData();
        float L = 0.0f;
        for (size_t i = 0; i < yd.size(); i++) L += gd[i] * yd[i];
        return L;
    };
    Tensor<float> analyticNoFlip = convolve(g, K, Mode::Full);          // claim: correct
    Tensor<float> analyticFlip   = convolve(g, K.flip(), Mode::Full);   // claim: wrong
    std::vector<float> numeric(X.getSize());
    for (size_t idx = 0; idx < (size_t)X.getSize(); idx++) {
        Tensor<float> Xp = X, Xm = X;
        Xp.getData()[idx] += eps;
        Xm.getData()[idx] -= eps;
        numeric[idx] = (computeLoss(Xp) - computeLoss(Xm)) / (2 * eps);
    }
    const auto& noFlip = analyticNoFlip.getData();
    const auto& flip   = analyticFlip.getData();
    float maxDiffNoFlip = 0.0f, maxDiffFlip = 0.0f;
    for (size_t i = 0; i < numeric.size(); i++) {
        maxDiffNoFlip = std::max(maxDiffNoFlip, std::abs(numeric[i] - noFlip[i]));
        maxDiffFlip   = std::max(maxDiffFlip,   std::abs(numeric[i] - flip[i]));
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "max diff (no flip): " << maxDiffNoFlip << "\n";
    std::cout << "max diff (flip):    " << maxDiffFlip << "\n\n";

    std::cout << "numeric   |  no-flip  |  flip\n";
    for (size_t i = 0; i < numeric.size(); i++)
        std::cout << std::setw(8) << numeric[i] << "  "
                  << std::setw(8) << noFlip[i] << "  "
                  << std::setw(8) << flip[i] << "\n";

    std::cout << "\n" << (maxDiffNoFlip < maxDiffFlip? "no flip passes yayaya" : "no-flip fails ggs");
}

template <typename T>
void printSample(const typename CNN<T> ::Dataset& data, size_t idx) {
    const auto& [image, label] = data[idx];
    const auto& pixelData = image[0].getData();
    const long long H = image[0].getShape()[0];
    const long long W = image[0].getShape()[1];
    std::cout << "Label: " << static_cast<char>('A' + label)   << "\n";
    for (long long r = 0; r < H; r++) {
        for (long long c = 0; c < W; c++) {
            T v = pixelData[r * W + c];
            std::cout << (v > 0.5f ? '#' : (v > 0.2f ? '.' : ' '));
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}
void benchmarkConv() {
    ConvLayer<float> conv(1, 8, 3);

    std::vector<Tensor<float>> input;
    std::vector<float> d(28*28);
    for (int i = 0; i < 28*28; i++) d[i] = 0.1f * (i % 10);
    input.emplace_back(std::vector<long long>{28,28}, d);

    const int trials = 50;

    // forward timing
    auto fStart = std::chrono::high_resolution_clock::now();
    std::vector<Tensor<float>> out;
    for (int i = 0; i < trials; i++)
        out = conv.forward(input);
    auto fEnd = std::chrono::high_resolution_clock::now();
    double fAvgUs = std::chrono::duration_cast<std::chrono::microseconds>(fEnd - fStart).count() / (double)trials;
    std::cout << "[conv1 28x28] avg forward: " << fAvgUs << " us (" << fAvgUs/1000.0 << " ms)\n";

    // backward timing
    std::vector<Tensor<float>> gradOut;
    for (const auto& o : out) {
        std::vector<float> gd(o.getSize(), 0.1f);
        gradOut.emplace_back(o.getShape(), gd);
    }
    conv.forward(input);   // populate cache once
    auto bStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; i++)
        conv.backward(gradOut);
    auto bEnd = std::chrono::high_resolution_clock::now();
    double bAvgUs = std::chrono::duration_cast<std::chrono::microseconds>(bEnd - bStart).count() / (double)trials;
    std::cout << "[conv1 28x28] avg backward: " << bAvgUs << " us (" << bAvgUs/1000.0 << " ms)\n";
}
void train(){
    std::string savePath = "../weights/";
    CNN<float> cnn;
    cnn.add(std::make_unique<ConvLayer<float>>(1, 8, 3));
    cnn.add(std::make_unique<maxPoolLayer<float>>(2));
    cnn.add(std::make_unique<ConvLayer<float>>(8, 16, 3));
    cnn.add(std::make_unique<maxPoolLayer<float>>(2));
    cnn.add(std::make_unique<FCLayer<float>>(16*5*5, 128, true));
    cnn.add(std::make_unique<FCLayer<float>>(128, 26, false));
    bool train;
    std::cout<<"1 for train 0 for not train";
    std::cin>>train;
    if (train) {
        auto trainData = loadEMNIST<float>(
                "../data/emnist-letters-train-images-idx3-ubyte/emnist-letters-train-images-idx3-ubyte",
                "../data/emnist-letters-train-labels-idx1-ubyte/emnist-letters-train-labels-idx1-ubyte",
                -1);

        auto testData = loadEMNIST<float>(
                "../data/emnist-letters-test-images-idx3-ubyte/emnist-letters-test-images-idx3-ubyte",
                "../data/emnist-letters-test-labels-idx1-ubyte/emnist-letters-test-labels-idx1-ubyte",
                -1);

        std::ifstream check(savePath + "weights.bin", std::ios::binary);
        if (check.good()) {
            check.close();
            cnn.load(savePath);
        } else {
            std::cout << "no existing weights check path"<<nl;
        }

        //train and watch loss go down btu lets see how it goes
        std::cout << "===========TRAIN=========" << nl;
        auto start = std::chrono::high_resolution_clock::now();
        cnn.train(trainData, 20,savePath, &testData);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "8 Epochs took: "
                  << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
                  << " s\n";

        std::cout << "per letter accuracy:"<<nl;
        // per letter breakdown
        cnn.perLetterAccuracy(testData);
    }else {
        auto testData = loadEMNIST<float>(
                "../data/emnist-letters-test-images-idx3-ubyte/emnist-letters-test-images-idx3-ubyte",
                "../data/emnist-letters-test-labels-idx1-ubyte/emnist-letters-test-labels-idx1-ubyte",
                500);
        cnn.load(savePath);
        std::cout<<"per letter accuracy:"<<nl;
        cnn.perLetterAccuracy(testData);
    }
}
int main() {
    auto data = loadBalancedDataset<float>(
            "../../data/finetune",
            "../data/emnist-letters-train-images-idx3-ubyte",
            "../data/emnist-letters-train-labels-idx1-ubyte",
            15,
            25
    );

}
#include <ApplicationServices/ApplicationServices.h>
#include "../../include/cv.hpp"
#include "../../include/engine.hpp"
#include "../../include/DAT.hpp"
#include "../../include/cnn.hpp"
#include "../../include/imgfx.hpp"
#include "../../include/move.hpp"
#include <thread>
#include <chrono> 


void dumpTensorAscii(Tensor<float>& t, int rows = 28, int cols = 28){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            float v = t[i][j];
            std::cout << (v > 0.5f ? '#' : '.');
        }
        std::cout << '\n';
    }
}

int main(){
    CNN<float> cnn;
    cnn.add(std::make_unique<ConvLayer<float>>(1, 8, 3));
    cnn.add(std::make_unique<maxPoolLayer<float>>(2));
    cnn.add(std::make_unique<ConvLayer<float>>(8, 16, 3));
    cnn.add(std::make_unique<maxPoolLayer<float>>(2));
    cnn.add(std::make_unique<FCLayer<float>>(16*5*5, 128, true));
    cnn.add(std::make_unique<FCLayer<float>>(128, 26, false));
    cnn.load("../../models/FineTuned/");
    unsigned int saveNum = 200;
    while(true){
        int t;
        std::cin>>t;
        if (t <= 0) break;
        if(t>0){
            auto temp = wrapperGetLetters("iPhone Mirroring");
            std::vector<cv::Mat> letters = temp.first;
            saveLetters(letters, saveNum,"../../images");
            saveNum+=16;
            //get all tensors of letters
            std::vector<std::vector<Tensor<float>>> imgTensors;
            for(const auto& mat : letters) {
                imgTensors.push_back(matToGrayTensor<float>(mat));
            }

            std::vector<std::pair<int, int>> coords = temp.second; //pixel coordinates for each thing so to play
            std::vector<std::vector<char>> grid(4, std::vector<char>(4));
            for(size_t x{}; x < 4; x++){
                for(size_t y{}; y < 4; y++){
                    size_t pos = x*4 + y;
                    int let = cnn.predict(imgTensors[pos]);
                    grid[x][y] = 'A'+let;
                    std::cout<<grid[x][y] << " ";
                }
                std::cout<<'\n';
            }
            //paths in mp[word]
            play(grid);
            std::cout<<"Found words:\n";
            words.print();
            while(!words.empty()){
                std::cout<<"Paths:\n";
                std::string curWord = words.popFront();
                std::cout<<curWord<<": ";
                const auto a = mp[curWord];
                for(const std::pair<int, int>& coord: a){
                    std::cout<<'{'<<coord.first <<", " <<coord.second<<'}'<<' ';
                }
                movePath(a, coords);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                std::cout<<"\n";
                //hehehehehe ggs it works
            }

        }

    }
}


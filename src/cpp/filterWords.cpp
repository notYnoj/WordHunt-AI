#include <iostream>
#include <fstream> 
#include <string>
#include <vector>

int main(){
    std::ifstream Input("../data/words.txt");
    std::ofstream Output("../data/dict.txt");
    std::string buffer;
    if (!Input.is_open()) {
        std::cerr << "Error: Could not open input file. Check if ../data/ exists." << std::endl;
        return 1; 
    }
    if (!Output.is_open()) {
        std::cerr << "Error: Could not open output file." << std::endl;
        return 1;
    }
    while(getline(Input, buffer)){
        if (!buffer.empty() && buffer.back() == '\r') {
            buffer.pop_back();
        }
        if(buffer.size() <= 16 && buffer.size() >= 3){
            Output<<buffer<<"\n";
        }
    }
    Input.close();
    Output.close();
}
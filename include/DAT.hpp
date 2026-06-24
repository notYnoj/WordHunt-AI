#pragma once
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>

class DAT {
private:
    std::vector<int> offset;
    std::vector<int> parent;
    void resize(int n);
public:
    DAT();
    DAT(std::string s);
    int toCode(char c);
    void insert(std::string s);
    bool search(std::string s);
    bool pref(std::string s);
    void clean();
    void save(const std::string& file);
    bool load(const std::string& filename);
};
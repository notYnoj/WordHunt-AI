#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <set>
#include <map>
#include "DAT.hpp"

extern std::vector<int> dir;
extern std::vector<std::pair<int, int>> dydx;
extern std::vector<std::vector<char>> grid;
extern std::map<std::string, std::vector<std::pair<int, int>>> mp;
extern std::mutex mp_mtx;
extern DAT trie;

template <typename T>
std::pair<T, T> operator+(std::pair<T,T> a, std::pair<T,T> b){
    return std::make_pair(a.first + b.first, a.second + b.second);
}

struct cmp{
    bool operator() (const std::string& a, const std::string& b){
        return a.size() > b.size();
    }
};

struct cmp2{
    bool operator() (const std::string& a, const std::string& b){
        return a.size() < b.size();
    }
};

class DEPrioritySet{
private:
    std::priority_queue<std::string, std::vector<std::string>, cmp2> maxpq;
    std::priority_queue<std::string, std::vector<std::string>, cmp> minpq;
    std::set<std::string> st;
    std::mutex mtx;
public:
    DEPrioritySet();
    bool empty();
    void push(const std::string& s);
    void remove(const std::string& s);
    std::string popFront();
    std::string popBack();
    void print();
};

extern DEPrioritySet words;

void dfs(std::pair<int, int> node, std::vector<std::pair<int, int>>& path, std::string& cur, std::vector<bool>& seen);
void play(std::vector<std::vector<char>>& g);
std::vector<std::vector<std::pair<int, int>>> getPaths();
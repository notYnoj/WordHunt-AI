#include "../../include/engine.hpp"

std::vector<int> dir = {0, 1, 2, 3};
std::vector<std::pair<int, int>> dydx = {{-1,0}, {0,-1}, {1,0}, {0,1}, {-1,-1}, {-1, 1}, {1, -1}, {1,1}};
std::vector<std::vector<char>> grid;
std::map<std::string, std::vector<std::pair<int, int>>> mp;  //a map from string to the path
std::mutex mp_mtx; //bro i got race errors LOL
DAT trie("../../data/trie/trieCLEAN.bin");

DEPrioritySet::DEPrioritySet(){

}
bool DEPrioritySet::empty(){
    return st.empty();
}
void DEPrioritySet::push(const std::string& s){
    std::lock_guard<std::mutex> lock(mtx);
    if(st.contains(s)){
        return;
    }
    st.insert(s);
    maxpq.push(s);
    minpq.push(s);
}
    
void DEPrioritySet::remove(const std::string& s){
    std::lock_guard<std::mutex> lock(mtx);
    auto it = st.find(s);
    if(it != st.end()){
        st.erase(it);
    }
}
    
std::string DEPrioritySet::popFront(){
    std::lock_guard<std::mutex> lock(mtx);
    while(!maxpq.empty()){
        if(st.contains(maxpq.top())){
            std::string result = maxpq.top();
            maxpq.pop();
            st.erase(result);
            return result;
        }
        maxpq.pop();
    }
    return "";
}

std::string DEPrioritySet::popBack(){
    std::lock_guard<std::mutex> lock(mtx);
    while(!minpq.empty()){
        if(st.contains(minpq.top())){
            std::string result = minpq.top();
            minpq.pop();
            st.erase(result);
            return result;
        }
        minpq.pop();
    }
    return "";
}

void DEPrioritySet::print(){
    std::vector<std::string> temp;
    while(!maxpq.empty()){
        std::string sTemp = popFront();
        std::cout<<sTemp<<' ';
        temp.push_back(sTemp);
    }
    std::cout<<'\n';
    for(const std::string& s : temp){
        push(s);
    }
}

DEPrioritySet words;

void dfs(std::pair<int, int> node, std::vector<std::pair<int, int>>& path, std::string& cur, std::vector<bool>& seen){
    if(trie.search(cur)){
        words.push(cur);
        std::lock_guard<std::mutex> lock(mp_mtx);
        mp[cur] = path;
    }
    seen[node.first * 4 + node.second] = true; 
    for(const std::pair<int, int>& step : dydx){
        std::pair<int, int> newNode = node + step;
        if(newNode.first > 3 || newNode.second > 3 || newNode.first < 0 || newNode.second < 0 || seen[newNode.first * 4 + newNode.second]){
            continue;
        }else{
            path.push_back(newNode);
            cur += grid[newNode.first][newNode.second];
            dfs(newNode, path, cur, seen);
            cur.pop_back();
            path.pop_back();
        }

    }
    seen[node.first * 4 + node.second] = false;
}


void play(std::vector<std::vector<char>>& g){
    /* Note that a word that starts at 0,0 can never be the same as a word that starts at 0,1
    We can break apart the workload across different tiles in our grid.
    Create parallel threads that each start at different points in the grid
    */
    std::vector<std::thread> threads;
    std::vector<std::vector<std::pair<int,int>>> paths(16);
    std::vector<std::vector<bool>> seens(16, std::vector<bool>(16, false));
    std::vector<std::string> curs(16, "");
    grid = std::move(g);

    for(int i: dir){
        for(int j: dir){
            std::vector<std::pair<int, int>> path;
            threads.push_back(std::thread(dfs, std::make_pair(i,j), std::ref(paths[i * 4 + j]), std::ref(curs[i * 4 + j]), std::ref(seens[i * 4 + j])));
        }
    }
    for(std::thread& t: threads){
        t.join();
    }
}

std::vector<std::vector<std::pair<int, int>>> getPaths(){
    std::vector<std::vector<std::pair<int, int>>> result;
    for(const auto& i: mp){
        result.push_back(i.second);
    }
    return result;
}
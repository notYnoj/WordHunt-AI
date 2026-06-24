#include "../../include/DAT.hpp"

DAT::DAT(){
    offset.resize(1500000, 0);
    parent.resize(1500000, -1);
    offset[0] = 1;
    parent[0] = 0;
}

DAT::DAT(std::string s){
    if(!DAT::load(s)){
        offset.resize(1500000, 0);
        parent.resize(1500000, -1);
        offset[0] = 1;
        parent[0] = 0;
    }
}

void DAT::resize(int n){
    if(offset.size() < n){
        offset.resize(n + 1000, 0);
        parent.resize(n+1000, -1);
    }
}

int DAT::toCode(char c){
    return(c == '#' ? 27 : c - 'A' + 1);
}

void DAT::insert(std::string s){
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return std::toupper(c);
    });
    s+="#";
    int root = 0;
    for(size_t i{0}; i < s.size(); i++){
        int target = offset[root] + toCode(s[i]);
        if (target >= (int)parent.size()) {
            resize(target);
        }
        //Not Open
        if(parent[target] != -1 && parent[target] != root){
            int targetParent = parent[target];
            std::vector<int> c1, c2;
            //count children of root + targetparent
            //whcihever one has less we will move those
            //move the children to a new place. Then update offset of parent
            //Also update any grandchildren to new index
            for(int child = 1; child <= 27; child++){
                int pChild = offset[root] + child;
                if(pChild > target){
                    //just make sure there is space so we dont have to check again
                    resize(pChild);
                }
                if(parent[pChild] == root){
                    c1.push_back(child);
                }
            }
            //c1 = cur movement c2 = squatter
            int newOffset;
            int oldOffset;
            int root2;
            std::vector<int> lookat;
            newOffset = offset[root];
            oldOffset = newOffset;
            root2 = root;
            lookat = c1;
            while(true){
                newOffset++;
                if(parent.size() <= newOffset){
                    resize(newOffset+27); //js make sure tehre is space
                }
                bool flag = true;
                for(int j: lookat){
                    //for letter if there is not space breka
                    if(parent[newOffset + j] != -1){
                        flag = false;
                        break;
                    }
                }
                if(flag){
                    if (parent[newOffset + toCode(s[i])] != -1) flag = false;
                }
                if(flag){
                    //found a valid position, begin moving
                    offset[root2] = newOffset;
                    for(int j : lookat){
                        offset[newOffset + j] = offset[oldOffset + j];
                        parent[newOffset + j] = root2;
                        if (offset[oldOffset + j] > 0) {
                            for(int child = 1; child <= 27; child++){
                                int gChild = offset[oldOffset + j] + child;
                                if(parent[gChild] == oldOffset + j){
                                    parent[gChild] = newOffset + j;
                                }
                            }
                        }
                        //dont ened this memorya nymore actualyl
                        offset[oldOffset + j] = 0;
                        parent[oldOffset + j] = -1;
                    }
                    break;
                }

            }
        }
        target = offset[root] + toCode(s[i]);
        //if new node just slap it back here
        if (parent[target] == -1) {
            parent[target] = root;
            offset[target] = 1;
        }
        root = target;
    }
}

bool DAT::search(std::string s){
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){
        return std::toupper(c);
    });

    s+="#";
    return pref(s);
}

bool DAT::pref(std::string s){
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){
        return std::toupper(c);
    });
    //js move in the tree
    int root = 0;
    for(size_t i{0}; i < s.size(); i++){
        int d = offset[root] + toCode(s[i]);
        if(d >= (int)parent.size() || d < 0 || parent[d] != root){
            return false;
        }
        root = d;
    }
    return true;
}

void DAT::clean(){
    //get rid of any extra space for efficencey
    int cnt = 0;
    for(int i = parent.size()-1; i>=1; i--){
        if(parent[i] == -1) cnt++;
        else break;
    }
    parent.resize(parent.size() - cnt);
    offset.resize(offset.size() - cnt);
}

void DAT::save(const std::string& file){
    //write to a binary
    std::ofstream out(file, std::ios::binary);
    size_t offsetSize = offset.size();
    size_t parentSize = parent.size();
    out.write(reinterpret_cast<const char*>(&offsetSize), sizeof(offsetSize));
    out.write(reinterpret_cast<const char*>(&parentSize), sizeof(parentSize));
    out.write(reinterpret_cast<const char*>(offset.data()), offsetSize * sizeof(int));
    out.write(reinterpret_cast<const char*>(parent.data()), parentSize * sizeof(int));
}

bool DAT::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;

    size_t offSize, parSize;
    in.read(reinterpret_cast<char*>(&offSize), sizeof(offSize));
    in.read(reinterpret_cast<char*>(&parSize), sizeof(parSize));
    offset.resize(offSize);
    parent.resize(parSize);
    in.read(reinterpret_cast<char*>(offset.data()), offSize * sizeof(int));
    in.read(reinterpret_cast<char*>(parent.data()), parSize * sizeof(int));
    //loaded fine
    return in.good();
}
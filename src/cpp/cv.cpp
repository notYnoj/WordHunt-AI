#include "../../include/cv.hpp"
#include "../../include/appFocus.h"
#include "../../include/imgfx.hpp"

void listAllWindows(){
    CFArrayRef windowList = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID);

    if(!windowList){
        std::cerr << "No windows. but how... ur running this..." << std::endl;
        return;
    }

    CFIndex windowCount = CFArrayGetCount(windowList);

    for(CFIndex i = 0; i < windowCount; i++){
        CFDictionaryRef windowInfo = (CFDictionaryRef) CFArrayGetValueAtIndex(windowList, i);
        CFStringRef windowName = (CFStringRef)CFDictionaryGetValue(windowInfo, kCGWindowOwnerName);
        
        if(windowName){
            char buffer[256];
            CFStringGetCString(windowName, buffer, sizeof(buffer), kCFStringEncodingUTF8);
            std::cout << i << ": " << buffer << std::endl;
        }
    }
        CFRelease(windowList);
}

std::pair<std::pair<int, int>, std::pair<int, int>> getAppCoordinates(const std::string& app){

    CFArrayRef windowList = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID);

    if(!windowList){
        std::cerr << "No windows" << std::endl;
        return {{-1, -1}, {-1, -1}};
    }

    CFStringRef targetApp = CFStringCreateWithCString(kCFAllocatorDefault, app.c_str(), kCFStringEncodingUTF8);


    std::pair<std::pair<int, int>, std::pair<int, int>> ret = {{-1, -1}, {-1, -1}};
    int area = 0;
    
    for(CFIndex i = 0; i < CFArrayGetCount(windowList); i++){
        CFDictionaryRef windowInfo = (CFDictionaryRef) CFArrayGetValueAtIndex(windowList, i);
        CFStringRef windowName = (CFStringRef)CFDictionaryGetValue(windowInfo, kCGWindowOwnerName);

        CFNumberRef windowLayer = (CFNumberRef)CFDictionaryGetValue(windowInfo, kCGWindowLayer);
        int layer = -1;
        if(windowLayer){
            CFNumberGetValue(windowLayer, kCFNumberIntType, &layer);
        }

        if(layer != 0) continue;
        if(windowName && CFStringCompare(windowName, targetApp, 0) == 0){
            CFDictionaryRef boundsDictionary = (CFDictionaryRef)CFDictionaryGetValue(
                windowInfo,
                kCGWindowBounds
            );
            
            if (boundsDictionary) {
                CGRect windowBounds;
                if (CGRectMakeWithDictionaryRepresentation(boundsDictionary, &windowBounds)) {
                    if(windowBounds.size.width * windowBounds.size.height >= area){
                        ret.first.first = (int)windowBounds.origin.x;
                        ret.first.second = (int)windowBounds.origin.y;
                        ret.second.first = (int)(windowBounds.origin.x + windowBounds.size.width);
                        ret.second.second = (int)(windowBounds.origin.y + windowBounds.size.height);
                        area = windowBounds.size.width * windowBounds.size.height;

                    }
                }
            }
        }
    }

    CFRelease(targetApp);
    CFRelease(windowList);
    if(ret.first.first == -1){
        std::cerr << "Window " << app <<" does not exist"<<std::endl;
    }

    return ret;
}

cv::Mat getSnapShot(const std::pair<int, int>& x1y1, const std::pair<int, int>& x2y2, const std::string& app){
    bringAppToForeground(app.c_str());
    usleep(500000);

    int width = x2y2.first - x1y1.first;
    int height = x2y2.second - x1y1.second;

    if(width <= 0 || height <= 0){
        std::cerr << "Invalid Rectangle in getSnapShot" << std::endl;
        return cv::Mat();
    }

    cv::Mat ret = captureRegion(x1y1.first, x1y1.second, width, height);

    if(ret.empty()){
        std::cerr << "Failed to capture region" << std::endl;
    }

    return ret;
}

cv::Mat grayScale(cv::Mat image){
    if(image.empty()){
        std::cerr << "Input image did not load" << std::endl;
        return cv::Mat();
    }

    cv::Mat ret;

    if(image.channels() == 1){
        ret = image.clone();
    }else{
        cv::cvtColor(image, ret, cv::COLOR_BGR2GRAY);
    }

    //std::cout << "Done" << std::endl;

    return ret;
}

cv::Mat findGrid(cv::Mat image){
    if(image.empty()){
        std::cerr << "Empty image" << std::endl;
        return cv::Mat();
    }
    cv::Mat img2 = image;
    if(img2.channels() != 1) img2 = grayScale(image);

    cv::Mat blurred;
    //blur for otsu to work better (sharp shi aka high frequency noise kinda fucks w/ it smooth w gausseian)
    cv::GaussianBlur(img2, blurred, cv::Size(3, 3), 0);
    //Creates black and white cut off w threshold
    cv::Mat binary;
    cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    //baiscally kernel forward prop
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat morphed;
    //morph (aka dilate then errode in order to connect shi)
    cv::morphologyEx(binary, morphed, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

    //canny is just canny
    cv::Mat edges;
    cv::Canny(morphed, edges, 50, 150);

    //Dilate so this time we can actually  connect lines
    cv::Mat ret;
    cv::Mat edgeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(edges, ret, edgeKernel, cv::Point(-1, -1), 2);

    return ret;
}

std::pair<std::vector<cv::Mat>, std::vector<cv::Rect>> extractLetters(cv::Mat processedGrid, cv::Mat image, double minPercent, double maxPercent){
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(processedGrid, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    double area = processedGrid.rows * processedGrid.cols;
    double minCellArea = area * minPercent; 
    double maxCellArea = area * maxPercent;
    std::vector<cv::Rect> cellRects;

    for(auto& i: contours){
        cv::Rect cur = cv::boundingRect(i);
        double area = cur.height * cur.width;
        double aspectRatio = (double)cur.width / cur.height;
        if(area >= minCellArea && area <= maxCellArea && aspectRatio >= 0.7 && aspectRatio <= 1.25){
            cellRects.push_back(cur);
        }
    }

    std::sort(cellRects.begin(), cellRects.end(), [](const cv::Rect& a, const cv::Rect& b) {
        if(std::abs(a.y - b.y) > 10) return a.y < b.y;
        return a.x < b.x;                         
    });

    std::vector<cv::Mat> ret;
    for(auto& rect : cellRects) {
        int inset = static_cast<int>(rect.width * 0.12);
        cv::Rect tight(rect.x + inset, rect.y + inset,
                       rect.width - 2*inset, rect.height - 2*inset);
        cv::Rect safeRect = tight & cv::Rect(0, 0, image.cols, image.rows);
        if(!safeRect.empty()) {
            ret.push_back(image(safeRect).clone());
        }
    }
    return {ret, cellRects};

}
/*
  compiled with: clang++ cv.cpp appfocus.mm -std=c++23 -o cv -O3 -march=native `pkg-config --cflags --libs opencv4` \
  -framework CoreGraphics \
  -framework CoreFoundation \
  -framework AppKit \
  -framework ScreenCaptureKit
*/
std::pair<std::vector<cv::Mat>, std::vector<std::pair<int, int>>> wrapperGetLetters(std::string s = "iPhone Mirroring"){
    listAllWindows();
    std::pair<std::pair<int, int>, std::pair<int, int>> a = getAppCoordinates(s);
    std::cout<<a.first.first<<' '<<a.first.second <<' ' << a.second.first <<' '<< a.second.second<<std::endl;
    cv::Mat snapshot = getSnapShot(a.first, a.second, s);
    cv::Mat processedGrid = findGrid(snapshot);
    double minPercent = 0.011;
    double maxPercent = 0.15;
    auto i = extractLetters(processedGrid, snapshot, minPercent, maxPercent);
    /*while(true){
        auto i = extractLetters(processedGrid, snapshot, minPercent, maxPercent);
        if(i.size() >= 16){
            minPercent += 0.001;
        }
        if(i.size() == 16){
            std::cout << minPercent << maxPercent;
            break;
        }
        if(minPercent >= maxPercent){
            std::cout << "It has to be greater" << std::endl;
            break;
        }
    }
    */
    //create a vector of cell coordinates 1 = x,y 2 = x2,y2
    auto ret = i.first;
    auto rects = i.second;
    std::vector<std::pair<int, int>> ret2;
    for(auto j : rects){
        int x = a.first.first + j.x + j.width/2;
        int y = a.first.second + j.y + j.height/2;
        ret2.push_back({x,y});
    }

    std::vector<cv::Mat> processedLetters;
    for(const auto& mat : ret){
        cv::Mat inverted;
        cv::bitwise_not(mat, inverted);
        //grayscale everything + turn into correct size
        processedLetters.push_back(binarizeLetter(toGray(resizeSquare(inverted, 28))));
    }

    return {processedLetters, ret2};
}

void saveLetters(const std::vector<cv::Mat>& letters, unsigned int& t, const std::string& outDir){
    namespace fs = std::filesystem;
    fs::path dir(outDir);
    if(!fs::exists(dir)){
        fs::create_directories(dir);
    }
    for(size_t idx = 0; idx < letters.size(); idx++){
        std::string filename = (dir / ("letter_" + std::to_string(t) + ".png")).string();
        t++;
        if(!cv::imwrite(filename, letters[idx])){
            std::cerr << "Failed to write " << filename << std::endl;
        }
    }
}

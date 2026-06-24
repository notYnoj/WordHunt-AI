#include "../../include/move.hpp"
#include <ApplicationServices/ApplicationServices.h>
#include <thread>
#include <chrono>

void movePath(const std::vector<std::pair<int, int>>& path, const std::vector<std::pair<int, int>>& coords){
    std::vector<std::pair<int, int>> posPath;
    for(auto i : path){
        posPath.push_back(coords[i.first * 4 + i.second]);
    }
    if(posPath.empty()) return;
    CGPoint start = CGPointMake(posPath[0].first, posPath[0].second);
    //go down first move there then drag then release at end
    CGEventRef mouseDown = CGEventCreateMouseEvent(nullptr, kCGEventLeftMouseDown, start, kCGMouseButtonLeft);
    CGEventPost(kCGHIDEventTap, mouseDown);
    CFRelease(mouseDown);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    for(size_t i = 1; i < posPath.size(); i++){
        CGPoint p = CGPointMake(posPath[i].first, posPath[i].second);
        CGEventRef drag = CGEventCreateMouseEvent(nullptr, kCGEventLeftMouseDragged, p, kCGMouseButtonLeft);
        CGEventPost(kCGHIDEventTap, drag);
        CFRelease(drag);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    CGPoint end = CGPointMake(posPath.back().first, posPath.back().second);
    CGEventRef mouseUp = CGEventCreateMouseEvent(nullptr, kCGEventLeftMouseUp, end, kCGMouseButtonLeft);
    CGEventPost(kCGHIDEventTap, mouseUp);
    CFRelease(mouseUp);
}
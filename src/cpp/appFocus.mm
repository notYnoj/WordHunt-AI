#import "../../include/appFocus.h"
#import <AppKit/AppKit.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#import <CoreGraphics/CoreGraphics.h>

void bringAppToForeground(const char* appName) {
    @autoreleasepool {
        NSString* name = [NSString stringWithUTF8String:appName];
        NSArray* allApps = [[NSWorkspace sharedWorkspace] runningApplications];
        for (NSRunningApplication* app in allApps) {
            if ([app.localizedName isEqualToString:name]) {
                [app activateWithOptions:0];
                break;
            }
        }
    }
}

cv::Mat captureRegion(int x, int y, int width, int height) {

    __block cv::Mat result;
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

    [SCShareableContent getShareableContentWithCompletionHandler:^(SCShareableContent* content, NSError* error) {
        if (error) {
            NSLog(@"Error getting shareable content: %@", error);
            dispatch_semaphore_signal(semaphore);
            return;
        }

        SCDisplay* targetDisplay = nil;
        for (SCDisplay* display in content.displays) {
            if (display.displayID == CGMainDisplayID()) {
                targetDisplay = display;
                break;
            }
        }

        if (!targetDisplay) {
            NSLog(@"No display found");
            dispatch_semaphore_signal(semaphore);
            return;
        }

        SCContentFilter* filter = [[SCContentFilter alloc] initWithDisplay:targetDisplay excludingWindows:@[]];

        SCStreamConfiguration* config = [[SCStreamConfiguration alloc] init];
        config.width = targetDisplay.width;
        config.height = targetDisplay.height;
        config.pixelFormat = kCVPixelFormatType_32BGRA;
        config.showsCursor = NO;

        [SCScreenshotManager captureImageWithFilter:filter 
                                      configuration:config 
                              completionHandler:^(CGImageRef cgImage, NSError* err) {
            if (err || !cgImage) {
                NSLog(@"Screenshot error: %@", err);
                dispatch_semaphore_signal(semaphore);
                return;
            }

            CGRect cropRect = CGRectMake(x, y, width, height);
            CGImageRef cropped = CGImageCreateWithImageInRect(cgImage, cropRect);

            if (!cropped) {
                NSLog(@"Failed to crop image");
                dispatch_semaphore_signal(semaphore);
                return;
            }

            size_t imgWidth  = CGImageGetWidth(cropped);
            size_t imgHeight = CGImageGetHeight(cropped);
            size_t bytesPerRow = CGImageGetBytesPerRow(cropped);  // ADD THIS BECAUSE OF THE FUCK ASS TEAR? WHAT THE FUCK

            CFDataRef imageData = CGDataProviderCopyData(CGImageGetDataProvider(cropped));
            if (imageData) {
                const UInt8* pixels = CFDataGetBytePtr(imageData);
                cv::Mat mat(imgHeight, imgWidth, CV_8UC4, (void*)pixels, bytesPerRow);
                result = mat.clone();
                cv::cvtColor(result, result, cv::COLOR_BGRA2RGB);
                CFRelease(imageData);
            }

            CGImageRelease(cropped);
            dispatch_semaphore_signal(semaphore);
        }];
    }];

    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
    return result;
}
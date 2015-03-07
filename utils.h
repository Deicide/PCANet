#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED
#include<opencv2/opencv.hpp>
#include<vector>
using namespace std;
using namespace cv;
#define IMG_DATA_LENGTH 3072
#define IMG_SIZE 32
#define CHANNELS 3
#define PATCH_SIZE 5

typedef struct{
    int numStages;
    vector<int > patchSize;
    vector<int > numFilters;
    vector<int > histBlockSize;
    float blkOverlapRatio;
    vector<int > pyramid;
}PCANet;

typedef struct{
    vector<int> block_idx;
    cv::Mat feature;
}HashingResult;

typedef struct{
    cv::Mat label;
    cv::Mat feature;
    vector<cv::Mat> filters;//each row is a PCA filter
}PCATrainResult;


#endif // UTILS_H_INCLUDED

// utils.h


#pragma once

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


double getPSNR(const Mat& I1, const Mat& I2);

Scalar getMSSIM(const Mat& i1, const Mat& i2);

double calculateSSIM(const cv::Mat& i1, const cv::Mat& i2);

#endif // UTILS_H
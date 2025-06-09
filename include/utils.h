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
#include <fstream>
#include <filesystem> 

#include "transformer.h"

using namespace cv;
using namespace std;


double getPSNR(const Mat& I1, const Mat& I2);


double calculateSSIM(const cv::Mat& i1, const cv::Mat& i2);


std::string shrinkTypeToString(HaarTransformer::Shrinktype type);


void process_test_mode(std::string input_dir, std::string output_csv);

#endif // UTILS_H
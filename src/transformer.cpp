#include "transformer.h"

HaarTransformer::HaarTransformer() {
    haar_channels = std::vector<cv::Mat>(3);
    
    for (int i = 0; i < 3; ++i) {
        haar_channels[i] = cv::Mat(local_image.rows, local_image.cols, CV_32FC1);
    }
}


void HaarTransformer::procces_channels(){

    cv::split(local_image, splitted_channels);

    // Приведение каждого канала к float:
    for (auto& c : splitted_channels) {
        c.convertTo(c, CV_32FC1, 1.0 / 255.0);
    }

}


void HaarTransformer::convert_to_YCbCr() {
    cv::cvtColor(local_image, local_image, cv::COLOR_BGR2YCrCb);

}


void HaarTransformer::upload_image(std::string path_to_image) {

    local_image = cv::imread(path_to_image);
    if (local_image.empty()) {
        std::cerr << "Error: failed to load image!" << std::endl;
    }

}


void HaarTransformer::apply_Haar(int NIter) {
    for (int i = 0; i < 3; ++i) {
        haar_channels[i] = cv::Mat(local_image.rows, local_image.cols, CV_32FC1);
        cvHaarWavelet(splitted_channels[i], haar_channels[i], NIter);
    }

}


void HaarTransformer::forward_transform(int NIter) {

    // 1. Преобразование канала YCbCr(опционально)
    if (type == Transtype::CBrCr) convert_to_YCbCr();


    // 2. Деление изображения на каналы
    procces_channels();


    // 3. Преобразования Хаара
    apply_Haar(NIter);

}


cv::Mat HaarTransformer::backward_transform(int NIter) {
    
    for (int i = 0; i < 3; ++i) {
        apply_inv_Haar(haar_channels[i], splitted_channels[i], NIter, HaarTransformer::Shrinktype::NONE, 50);
    }

    for (auto& c : splitted_channels) {
        c.convertTo(c, CV_8UC1, 255.0);  // вернём 0–255
    }

    cv::merge(splitted_channels, out_image);

    cv::cvtColor(out_image, out_image, cv::COLOR_YCrCb2BGR);

    return out_image;

}


void HaarTransformer::apply_inv_Haar(cv::Mat& channel, cv::Mat& out_channel, int NIter, Shrinktype SHRINKAGE_TYPE = Shrinktype::NONE, float SHRINKAGE_T = 50)
{
    float c, dh, dv, dd;
    assert(channel.type() == CV_32FC1);
    assert(out_channel.type() == CV_32FC1);
    int width = channel.cols;
    int height = channel.rows;

    // NIter - number of iterations 
    for (int k = NIter;k > 0;k--)
    {
        for (int y = 0;y < (height >> k);y++)
        {
            for (int x = 0; x < (width >> k);x++)
            {
                c = channel.at<float>(y, x);
                dh = channel.at<float>(y, x + (width >> k));
                dv = channel.at<float>(y + (height >> k), x);
                dd = channel.at<float>(y + (height >> k), x + (width >> k));

                // (shrinkage)
                switch (SHRINKAGE_TYPE)
                {
                case Shrinktype::HARD:
                    dh = hard_shrink(dh, SHRINKAGE_T);
                    dv = hard_shrink(dv, SHRINKAGE_T);
                    dd = hard_shrink(dd, SHRINKAGE_T);
                    break;
                case Shrinktype::SOFT:
                    dh = soft_shrink(dh, SHRINKAGE_T);
                    dv = soft_shrink(dv, SHRINKAGE_T);
                    dd = soft_shrink(dd, SHRINKAGE_T);
                    break;
                case Shrinktype::GARROT:
                    dh = Garrot_shrink(dh, SHRINKAGE_T);
                    dv = Garrot_shrink(dv, SHRINKAGE_T);
                    dd = Garrot_shrink(dd, SHRINKAGE_T);
                    break;
                }

                //-------------------
                out_channel.at<float>(y * 2, x * 2) = 0.5 * (c + dh + dv + dd);
                out_channel.at<float>(y * 2, x * 2 + 1) = 0.5 * (c - dh + dv - dd);
                out_channel.at<float>(y * 2 + 1, x * 2) = 0.5 * (c + dh - dv - dd);
                out_channel.at<float>(y * 2 + 1, x * 2 + 1) = 0.5 * (c - dh - dv + dd);
            }
        }
        cv::Mat C = channel(cv::Rect(0, 0, width >> (k - 1), height >> (k - 1)));
        cv::Mat D = out_channel(cv::Rect(0, 0, width >> (k - 1), height >> (k - 1)));
        D.copyTo(C);
    }
}
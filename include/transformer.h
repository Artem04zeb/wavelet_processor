// haar_transform.h

#pragma once

#ifndef HAAR_TRANSFORM_H
#define HAAR_TRANSFORM_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

/**
 * @class HaarTransformer
 * @brief Класс для выполнения прямого и обратного 2D-преобразования Хаара на изображениях.
 */
class HaarTransformer {
public:
    /**
     * @brief Конструктор по умолчанию.
     */
    HaarTransformer();


    /**
     * @brief Выполняет прямое преобразование Хаара для каждого цветового канала изображения.
     * @param NIter Количество уровней (итераций) декомпозиции.
     */
    void forward_transform(int NIter);


    /**
     * @brief Выполняет обратное преобразование Хаара и восстанавливает изображение.
     * @param NIter Количество уровней (итераций), соответствующее прямому преобразованию.
     * @return Восстановленное изображение (cv::Mat).
     */
    cv::Mat backward_transform(int NIter);


    /**
     * @brief Разделяет изображение на цветовые каналы и применяет преобразование Хаара к каждому из них.
     */
    void procces_channels();

    
    /**
     * @brief Преобразует изображение из RGB в цветовое пространство YCbCr.
     */
    void convert_to_YCbCr();


    /**
     * @brief Загружает изображение из файла.
     * @param path_to_image Путь к файлу изображения.
     */
    void upload_image(std::string path_to_image);

    /**
     * @enum Transtype
     * @brief Тип используемого цветового пространства.
     */
    enum class Transtype : int {
        RGB,    ///< Обычное RGB
        CBrCr,  ///< Цветовое пространство YCbCr
        Gray    ///< Градации серого
    };


    /**
     * @enum Shrinktype
     * @brief Тип пороговой функции для подавления коэффициентов.
     */
    enum class Shrinktype : int {
        NONE,   ///< Без пороговой фильтрации
        HARD,   ///< Жёсткая пороговая фильтрация
        SOFT,   ///< Мягкая пороговая фильтрация
        GARROT  ///< Пороговая фильтрация Гаррота
    };


    /**
     * @brief Выполняет прямое преобразование Хаара и сохраняет результат.
     * @param NIter Количество уровней декомпозиции.
     */
    void apply_Haar(int NIter);


    /**
     * @brief Выполняет обратное преобразование Хаара для одного канала.
     * @param channel Исходный канал с коэффициентами Хаара.
     * @param out_channel Выходной канал (восстановленный).
     * @param NIter Количество уровней.
     * @param SHRINKAGE_TYPE Тип пороговой фильтрации.
     * @param SHRINKAGE_T Пороговое значение для фильтрации.
     */
    void apply_inv_Haar(cv::Mat& channel, cv::Mat& out_channel, int NIter, Shrinktype SHRINKAGE_TYPE, float SHRINKAGE_T);
    

private:
    /// @brief Исходное изображение.
    cv::Mat local_image;

    /// @brief Восстановленное изображение.
    cv::Mat out_image;

    /// @brief Вектор разложенных цветовых каналов.
    std::vector<cv::Mat> splitted_channels;

    /// @brief Вектор каналов после преобразования Хаара.
    std::vector<cv::Mat> haar_channels;

    /// @brief Цветовое пространство по умолчанию.  
    Transtype type = Transtype::CBrCr;

    /// @brief Максимальное количество уровней разложения по умолчанию
    int max_levels_ = 3;


    /**
     * @brief Выполняет прямое 2D-преобразование Хаара.
     * @param src Входной канал (тип CV_32FC1).
     * @param dst Выходной канал (тип CV_32FC1).
     * @param NIter Количество уровней декомпозиции.
     */
    void cvHaarWavelet(cv::Mat& src, cv::Mat& dst, int NIter)
    {
        float coef_c, coef_dh, coef_dv, coef_dd;
        assert(src.type() == CV_32FC1);
        assert(dst.type() == CV_32FC1);
        int width = src.cols;
        int height = src.rows;
        for (int k = 0;k < NIter; k++)
        {
            for (int y = 0; y < (height >> (k + 1)); y++)
            {
                for (int x = 0; x < (width >> (k + 1)); x++)
                {
                    // Чтение 2×2 блока
                    float a = src.at<float>(2 * y, 2 * x);
                    float b = src.at<float>(2 * y, 2 * x + 1);
                    float c = src.at<float>(2 * y + 1, 2 * x);
                    float d = src.at<float>(2 * y + 1, 2 * x + 1);

                    // Вычисление коэффициентов Хаара
                    coef_c = (a + b + c + d) * 0.5;
                    coef_dh = (a + c - b - d) * 0.5;
                    coef_dv = (a + b - c - d) * 0.5;
                    coef_dd = (a - b - c + d) * 0.5;

                    float half_width = width >> (k + 1);
                    float half_height = height >> (k + 1);

                    dst.at<float>(y, x) = coef_c;  // LL (приближённое значение)
                    dst.at<float>(y, x + half_width) = coef_dh;  // LH (горизонталь)
                    dst.at<float>(y + half_height, x) = coef_dv; // HL (вертикаль)
                    dst.at<float>(y + half_height, x + half_width) = coef_dd; // HH (диагональ)
                }
            }
            dst.copyTo(src);
        }
    }


    /**
     * @brief Возвращает знак числа.
     * @param x Входное число.
     * @return 1 если x > 0, -1 если x < 0, 0 если x == 0.
     */
    float sgn(float x)
    {
        if (x == 0) return 0;
        else if (x > 0) return 1;
        else return 1;
    }


    /**
     * @brief Мягкая пороговая фильтрация (soft thresholding).
     * @param d Коэффициент.
     * @param T Порог.
     * @return Отфильтрованное значение.
     */
    float soft_shrink(float d, float T)
    {
        if (fabs(d) > T) return sgn(d) * (fabs(d) - T);
        else return 0;
    }


    /**
     * @brief Жёсткая пороговая фильтрация (hard thresholding).
     * @param d Коэффициент.
     * @param T Порог.
     * @return Отфильтрованное значение.
     */
    float hard_shrink(float d, float T)
    {
        if (fabs(d) > T) return d;
        else return 0;
    }


    /**
     * @brief Пороговая фильтрация по Гарроту (Garrot thresholding).
     * @param d Коэффициент.
     * @param T Порог.
     * @return Отфильтрованное значение.
     */
    float Garrot_shrink(float d, float T)
    {
        if (fabs(d) > T) return d - ((T * T) / d);
        else return 0;
    }

};

#endif // HAAR_TRANSFORM_H
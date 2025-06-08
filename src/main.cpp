#include "transformer.h"


int main(int argc, char* argv[]) {
    // Настройки консоли
    setlocale(LC_CTYPE, "rus");
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    std::string inputPath = "D:\\Desktop\\wavelet_compression\\data\\cat.jpg";

    HaarTransformer trans;  // Создание объекта класса трансформер
    trans.upload_image(inputPath);  // 1. Загрузка изображения
    trans.forward_transform(3);  // 2. Трансформирование изображения

    cv::Mat output;
    output = trans.backward_transform(3);

    cv::imshow("test", output);
    cv::waitKey();

    return 0;
}
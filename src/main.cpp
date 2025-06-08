#include "transformer.h"


int main(int argc, char* argv[]) {
    // ��������� �������
    setlocale(LC_CTYPE, "rus");
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    std::string inputPath = "D:\\Desktop\\wavelet_compression\\data\\cat.jpg";

    HaarTransformer trans;  // �������� ������� ������ �����������
    trans.upload_image(inputPath);  // 1. �������� �����������
    trans.forward_transform(3);  // 2. ����������������� �����������

    cv::Mat output;
    output = trans.backward_transform(3);

    cv::imshow("test", output);
    cv::waitKey();

    return 0;
}
#include "utils.h"


double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}


double calculateSSIM(const cv::Mat& i1, const cv::Mat& i2) {
    const double C1 = 6.5025, C2 = 58.5225;
    cv::Mat I1, I2;
    i1.convertTo(I1, CV_32F);
    i2.convertTo(I2, CV_32F);

    cv::Mat I1_2 = I1.mul(I1);
    cv::Mat I2_2 = I2.mul(I2);
    cv::Mat I1_I2 = I1.mul(I2);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);

    return cv::mean(ssim_map)[0];
}


std::string shrinkTypeToString(HaarTransformer::Shrinktype type) {
    switch (type) {
    case HaarTransformer::Shrinktype::NONE:   return "NONE";
    case HaarTransformer::Shrinktype::HARD:   return "HARD";
    case HaarTransformer::Shrinktype::SOFT:   return "SOFT";
    case HaarTransformer::Shrinktype::GARROT: return "GARROT";
    default:                 return "UNKNOWN";
    }
}


void process_test_mode(std::string input_dir, std::string output_csv){
    namespace fs = std::filesystem;

    // Параметры для тестирования
    const std::vector<int> n_iter_values = { 2, 3, 4 };
    const std::vector<HaarTransformer::Shrinktype> shrink_types = {
        HaarTransformer::Shrinktype::NONE,
        HaarTransformer::Shrinktype::HARD,
        HaarTransformer::Shrinktype::SOFT,
        HaarTransformer::Shrinktype::GARROT
    };
    const std::vector<float> shrinkage_values = { 25.0f, 50.0f, 80.0f };

    // Открытие файла для записи результатов
    std::ofstream csv_file(output_csv);
    csv_file << "Filename,NIter,Shrinktype,Shrinkage,PSNR,SSIM\n";

    // Обработка каждого изображения в папке
    HaarTransformer trans;
    int total_processed = 0;

    // Обработка каждого изображения
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.path().extension() != ".png") continue;

        std::string filename = entry.path().filename().string();
        cv::Mat original = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (original.empty()) {
            std::cerr << "Error loading: " << filename << std::endl;
            continue;
        }

        // Тестирование всех комбинаций параметров
        for (int n_iter : n_iter_values) {
            for (HaarTransformer::Shrinktype shrink_type : shrink_types) {
                for (float shrinkage : shrinkage_values) {
                    try {
                        // Обработка изображения
                        trans.upload_image(entry.path().string());
                        trans.forward_transform(n_iter);
                        cv::Mat reconstructed = trans.backward_transform(
                            n_iter, shrink_type, shrinkage);

                        // Вычисление метрик
                        double psnr_val = getPSNR(original, reconstructed);
                        double ssim_val = calculateSSIM(original, reconstructed);

                        // Запись результатов
                        csv_file << filename << ","
                            << n_iter << ","
                            << shrinkTypeToString(shrink_type) << ","
                            << shrinkage << ","
                            << std::fixed << std::setprecision(4)
                            << psnr_val << ","
                            << ssim_val << "\n";

                        // Вывод прогресса
                        std::cout << "Processed " << filename
                            << " | NIter=" << n_iter
                            << " | Type=" << shrinkTypeToString(shrink_type)
                            << " | Shrink=" << shrinkage
                            << " | PSNR=" << psnr_val
                            << " | SSIM=" << ssim_val << std::endl;

                        total_processed++;
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error processing " << filename
                            << " with params (" << n_iter << ", "
                            << shrinkTypeToString(shrink_type) << ", "
                            << shrinkage << "): " << e.what() << std::endl;
                    }
                }
            }
        }
    }

    csv_file.close();
    std::cout << "\nProcessing complete. Total tests: " << total_processed
        << "\nResults saved to " << output_csv << std::endl;




    }
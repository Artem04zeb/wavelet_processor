#include "transformer.h"
#include "utils.h"
#include <filesystem> 
#include <fstream>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // Настройки консоли
    setlocale(LC_CTYPE, "rus");
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    
    // Настройка путей
    const std::string input_dir = "D:\\Desktop\\wavelet_compression\\data\\clic\\";
    const std::string output_csv = "D:\\Desktop\\wavelet_compression\\data\\processed\\results.csv";

    // Открытие файла для записи результатов
    std::ofstream csv_file(output_csv);
    csv_file << "Filename,PSNR,SSIM\n";

    // Обработка каждого изображения в папке
    HaarTransformer trans;
    int processed_count = 0;

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.path().extension() != ".png") continue;

        try {
            std::string filename = entry.path().filename().string();
            std::string full_path = entry.path().string();

            // Загрузка оригинального изображения
            cv::Mat original = cv::imread(full_path, cv::IMREAD_COLOR);
            if (original.empty()) {
                std::cerr << "Error loading: " << filename << std::endl;
                continue;
            }

            // Обработка изображения
            trans.upload_image(full_path);
            trans.forward_transform(3);
            cv::Mat reconstructed = trans.backward_transform(3);

            // Вычисление метрик
            double psnr_val = getPSNR(original, reconstructed);
            double ssim_val = calculateSSIM(original, reconstructed);

            // Запись результатов
            csv_file << filename << ","
                << std::fixed << std::setprecision(4)
                << psnr_val << ","
                << ssim_val << "\n";

            // Вывод в консоль для мониторинга
            std::cout << "Processed " << filename
                << " | PSNR: " << psnr_val
                << " | SSIM: " << ssim_val << std::endl;

            processed_count++;

        }
        catch (const std::exception& e) {
            std::cerr << "Error processing file: " << e.what() << std::endl;
        }
    }

    csv_file.close();
    std::cout << "\nProcessing complete. " << processed_count
        << " images processed. Results saved to " << output_csv << std::endl;


    return 0;
}
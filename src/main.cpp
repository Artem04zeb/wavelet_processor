#include "transformer.h"
#include "utils.h"

namespace fs = std::filesystem;


int main(int argc, char* argv[]) {
    // Настройки консоли
    setlocale(LC_CTYPE, "rus");
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    
    // Настройка путей
    //std::string input_dir = "D:\\Desktop\\wavelet_compression\\data\\clic\\";
    //std::string output_csv = "D:\\Desktop\\wavelet_compression\\data\\processed\\results.csv";

    // Проверка минимального количества аргументов
    if (argc < 2) {
        std::cerr << "Usage:\n"
            << "  Test mode: " << argv[0] << " test <input_dir> <output_csv>\n"
            << "  Work mode: " << argv[0] << " work <src_path> <dst_path> <NIter> <shrinktype> <shrinkage>\n";
        return 1;
    }

    std::string mode(argv[1]);

    if (mode == "test") {
        // Режим тестирования (пакетная обработка)
        if (argc != 4) {
            std::cerr << "Error: test mode requires 2 additional arguments\n"
                << "Usage: " << argv[0] << " test <input_dir> <output_csv>\n";
            return 1;
        }

        std::string input_dir(argv[2]);
        std::string output_csv(argv[3]);

        // Проверка существования папки
        if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
            std::cerr << "Error: input directory does not exist or is not a directory\n";
            return 1;
        }

        // Вызов тестового режима
        process_test_mode(input_dir, output_csv);

        std::cout << "Running in TEST mode\n"
            << "Input directory: " << input_dir << "\n"
            << "Output CSV: " << output_csv << std::endl;

    }
    else if (mode == "work") {
        // Режим обработки одного изображения
        if (argc != 7) {
            std::cerr << "Error: work mode requires 5 additional arguments\n"
                << "Usage: " << argv[0] << " work <src_path> <dst_path> <NIter> <shrinktype> <shrinkage>\n";
            return 1;
        }

        std::string src_path(argv[2]);
        std::string dst_path(argv[3]);
        int n_iter = std::stoi(argv[4]);
        std::string shrinktype_str(argv[5]);
        float shrinkage = std::stof(argv[6]);

        // Проверка существования файла
        if (!fs::exists(src_path)) {
            std::cerr << "Error: source file does not exist\n";
            return 1;
        }

        // Преобразование shrinktype из строки в enum
        HaarTransformer::Shrinktype shrinktype;
        if (shrinktype_str == "NONE") {
            shrinktype = HaarTransformer::Shrinktype::NONE;
        }
        else if (shrinktype_str == "HARD") {
            shrinktype = HaarTransformer::Shrinktype::HARD;
        }
        else if (shrinktype_str == "SOFT") {
            shrinktype = HaarTransformer::Shrinktype::SOFT;
        }
        else if (shrinktype_str == "GARROT") {
            shrinktype = HaarTransformer::Shrinktype::GARROT;
        }
        else {
            std::cerr << "Error: invalid shrinktype. Use NONE, HARD, SOFT or GARROT\n";
            return 1;
        }

        // Обработка изображения
        HaarTransformer trans;
        trans.upload_image(src_path);
        trans.forward_transform(n_iter);
        cv::Mat result = trans.backward_transform(n_iter, shrinktype, shrinkage);

        // Сохранение результата
        if (!cv::imwrite(dst_path, result)) {
            std::cerr << "Error: failed to save result image\n";
            return 1;
        }

        std::cout << "Successfully processed image:\n"
            << "  Source: " << src_path << "\n"
            << "  Result: " << dst_path << "\n"
            << "  Parameters: NIter=" << n_iter
            << ", Shrinktype=" << shrinktype_str
            << ", Shrinkage=" << shrinkage << std::endl;

    }
    else {
        std::cerr << "Error: unknown mode. Use 'test' or 'work'\n";
        return 1;
    }


    return 0;
}
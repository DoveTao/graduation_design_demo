#include <System.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> ReadLines(const std::string& path) {
    std::ifstream input(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line[0] != '#') {
            lines.push_back(line);
        }
    }
    return lines;
}

std::vector<double> ReadTimestamps(const std::string& path) {
    std::ifstream input(path);
    std::vector<double> values;
    double value = 0.0;
    while (input >> value) {
        values.push_back(value);
    }
    return values;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: orbslam3_mono_fisheye_runner VOCAB SETTINGS IMAGE_LIST TIMESTAMPS OUT_TUM\n";
        return 2;
    }

    const std::string vocab = argv[1];
    const std::string settings = argv[2];
    const std::string image_list_path = argv[3];
    const std::string timestamps_path = argv[4];
    const std::string output_path = argv[5];

    const std::vector<std::string> image_paths = ReadLines(image_list_path);
    const std::vector<double> timestamps = ReadTimestamps(timestamps_path);
    if (image_paths.empty() || image_paths.size() != timestamps.size()) {
        std::cerr << "Image/timestamp count mismatch: images=" << image_paths.size()
                  << " timestamps=" << timestamps.size() << "\n";
        return 3;
    }

    std::ofstream output(output_path);
    if (!output.is_open()) {
        std::cerr << "Failed to open trajectory output: " << output_path << "\n";
        return 4;
    }
    output << std::fixed << std::setprecision(9);

    ORB_SLAM3::System slam(vocab, settings, ORB_SLAM3::System::MONOCULAR, false);
    const float image_scale = slam.GetImageScale();

    int success_count = 0;
    int failure_count = 0;
    for (size_t i = 0; i < image_paths.size(); ++i) {
        cv::Mat image = cv::imread(image_paths[i], cv::IMREAD_UNCHANGED);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_paths[i] << "\n";
            ++failure_count;
            continue;
        }
        if (image_scale != 1.0f) {
            cv::resize(image, image, cv::Size(), image_scale, image_scale);
        }

        Sophus::SE3f tcw = slam.TrackMonocular(image, timestamps[i], std::vector<ORB_SLAM3::IMU::Point>(), image_paths[i]);
        const int state = slam.GetTrackingState();
        if (state == 2 || state == 5) {
            Sophus::SE3f twc = tcw.inverse();
            Eigen::Vector3f t = twc.translation();
            Eigen::Quaternionf q = twc.unit_quaternion();
            output << timestamps[i] << " "
                   << t.x() << " " << t.y() << " " << t.z() << " "
                   << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
            ++success_count;
        } else {
            ++failure_count;
        }
    }

    slam.Shutdown();
    output.close();
    std::cout << "ORB1c runner summary: frames=" << image_paths.size()
              << " success=" << success_count
              << " failure=" << failure_count
              << " output=" << output_path << "\n";
    return 0;
}

#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <numeric>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/highgui.hpp>

#include "benchmark.hpp"

/* OPTION I:
 * Use random images for testing. 
 *
 * OPTION II:
 * Use images in "data/images/img_n.jpg" where `n` varies from 0, 1, 2, 3, ....
 */
#define USE_RANDOM_IMAGES

constexpr auto default_batch_size = 1;

struct mask_type {
    int backend;
    int target;
};

struct config_type {
    std::string name;
    int backend;
    int target;
};

// select backend target combinations that you want to test
std::vector<config_type> backends = {
    //{"OCV CPU", cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU},
    //{"OCV OpenCL", cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL},
    //{"OCV OpenCL FP16", cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL_FP16},

    //{"IE CPU", cv::dnn::DNN_BACKEND_INFERENCE_ENGINE, cv::dnn::DNN_TARGET_CPU},

    {"CUDA FP32", cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA},
    {"CUDA FP16", cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16}
};

std::vector<cv::Mat> image_samples;

template <class T>
auto to_milliseconds(const T& duration) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration);
}

template <class T>
auto to_microseconds(const T& duration) {
    return std::chrono::duration_cast<std::chrono::microseconds>(duration);
}

struct perf_result_t
{
    using duration = std::chrono::microseconds;

    duration init_time;
    std::vector<duration> runtimes;
};

template <std::size_t BENCHMARK_RUNS, std::size_t WARMUP_RUNS>
auto run_network(
    const std::string& model, const std::string& config,
    const cv::Mat& blob,
    const std::vector<std::string>& output_names_,
    int backend, int target)
{
    auto net = cv::dnn::readNet(model, config);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    auto output_names = output_names_;
    if (output_names.empty())
        output_names = net.getUnconnectedOutLayersNames();

    std::vector<cv::Mat> output_mats;
    auto init_time = benchmark([&] {
        net.setInput(blob);
        net.forward(output_mats, output_names);
    });

    for(int i = 0; i < WARMUP_RUNS; i++)
    {
        net.setInput(blob);
        net.forward(output_mats, output_names);
    }

    perf_result_t result;
    result.init_time = init_time;
    result.runtimes.reserve(BENCHMARK_RUNS);
    
    for(int i = 0; i < BENCHMARK_RUNS; i++)
    {
        net.setInput(blob);
        auto inference_time = benchmark([&] {
            net.forward(output_mats, output_names);
        });

        result.runtimes.push_back(inference_time);
    }    

    return result;
}

void bench_network(
    const std::string& model, const std::string& config,
    cv::Size input_size,
    const std::vector<std::string>& output_names = {},
    int count = default_batch_size,
    std::vector<mask_type> mask = {})
{
#ifndef USE_RANDOM_IMAGES
    assert(count <= image_samples.size());
#endif
  
    std::vector<cv::Mat> images;
    for (int i = 0; i < count; i++)
    {
#ifdef USE_RANDOM_IMAGES
        cv::Mat image(input_size, CV_32FC3);
        cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        images.push_back(image);
#else
        images.push_back(image_samples[i]);
#endif
    }

    cv::Mat blob = cv::dnn::blobFromImages(images, 1.0f, input_size, 0.0f);

    for (auto c : backends) {
        auto backend = c.backend;
        auto target = c.target;

        bool skip = [backend, target, mask] {
            for (auto m : mask) {
                if (m.backend == backend && m.target == target)
                    return true;
                if (m.backend == backend && m.target == -1)
                    return true;
                if (m.backend == -1 && m.target == target)
                    return true;
            }

            return false;
        } ();

        if(skip)
            continue;

        try {
            constexpr int WARMUP_RUNS = 10;
            constexpr int BENCHMARK_RUNS = 100;
            auto result = run_network<BENCHMARK_RUNS, WARMUP_RUNS>(model, config, blob, output_names, backend, target);

            float init_time = to_microseconds(result.init_time).count() / 1000.0;

            std::vector<float> runtimes;
            for (auto r : result.runtimes)
                runtimes.push_back(to_microseconds(r).count() / 1000.0);

            auto sum = std::accumulate(std::begin(runtimes), std::end(runtimes), 0.0f);
            auto squared_sum = std::inner_product(std::begin(runtimes), std::end(runtimes), std::begin(runtimes), 0.0f);
            
            auto min = *std::min_element(std::begin(runtimes), std::end(runtimes));
            auto max = *std::max_element(std::begin(runtimes), std::end(runtimes));
            auto mean = sum / runtimes.size();
            auto stddev = std::sqrt(squared_sum / runtimes.size() - mean * mean);

            std::cout << '[' << c.name << "]" << '\n'
                        << "\tinit >> " << init_time << "ms" << '\n'
                        << "\tinference >> " << "min = " << min << "ms, max = " << max << "ms, mean = " << mean << "ms, stddev = " << stddev << "ms" << std::endl;
        } catch(const std::exception& ex) {
            std::cout << ex.what() << std::endl;
            return;
        }
    }

    std::cout << std::endl;
}

void bench_alexnet()
{
    std::cout << "BVLC AlexNet\n";
    bench_network("data/alexnet/deploy.prototxt", "data/alexnet/bvlc_alexnet.caffemodel", cv::Size(227, 227));
    std::cout << std::endl;
}

void bench_densenet121()
{
    std::cout << "DenseNet 121\n";
    bench_network("data/densenet121/DenseNet_121.prototxt", "data/densenet121/DenseNet_121.caffemodel", cv::Size(224, 224));
    std::cout << std::endl;
}

void bench_east_text_detection()
{
    std::cout << "East Text Detection\n";
    bench_network("data/east_text_detection/frozen_east_text_detection.pb", "", cv::Size(320, 320));
    std::cout << std::endl;
}

void bench_enet()
{
    std::cout << "ENet Cityscapes\n";
    bench_network("data/enet/model-cityscapes.net", "", cv::Size(512, 256), {}, 1);
    std::cout << std::endl;
}

void bench_fns_stary_night()
{
    std::cout << "FastNeuralStyle Stary Night\n";
    bench_network("data/fns_stary_night/fast_neural_style_eccv16_starry_night.t7", "", cv::Size(320, 240));
    std::cout << std::endl;
}

void bench_googlenet()
{
    std::cout << "BVLC GoogleNet\n";
    bench_network("data/googlenet/deploy.prototxt", "data/googlenet/bvlc_googlenet.caffemodel", cv::Size(224, 224));
    std::cout << std::endl;
}

void bench_inception_v2_faster_rcnn()
{
    std::cout << "Inception v2 Faster RCNN\n";
    bench_network("data/inception_v2_faster_rcnn/faster_rcnn_inception_v2_coco_2018_01_28.pb", "data/inception_v2_faster_rcnn/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt", cv::Size(800, 600), {}, default_batch_size,
    {
        {cv::dnn::DNN_BACKEND_INFERENCE_ENGINE, -1}
    });
    std::cout << std::endl;
}

void bench_inception_v2_mask_rcnn()
{
    std::cout << "Inception v2 Mask RCNN\n";
    bench_network("data/inception_v2_mask_rcnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt", "data/inception_v2_mask_rcnn/mask_rcnn_inception_v2_coco_2018_01_28.pb", cv::Size(1024, 1024), { "detection_out_final", "detection_masks"});
    std::cout << std::endl;
}

void bench_mobilenet_ssd()
{
    std::cout << "MobileNet SSD\n";
    bench_network("data/mobilenet_ssd/MobileNetSSD_deploy.prototxt", "data/mobilenet_ssd/MobileNetSSD_deploy.caffemodel", cv::Size(300, 300));
    std::cout << std::endl;
}

void bench_mobilenet_ssd_v1_coco()
{
    std::cout << "MobileNet SSD v1 Coco\n";
    bench_network("data/mobilenet_ssd_v1_coco_2017_11_17/ssd_mobilenet_v1_coco_2017_11_17.pb", "data/mobilenet_ssd_v1_coco_2017_11_17/ssd_mobilenet_v1_coco_2017_11_17.pbtxt", cv::Size(300, 300));
    std::cout << std::endl;
}

void bench_mobilenet_ssd_v2_coco()
{
    std::cout << "MobileNet SSD v2 Coco\n";
    bench_network("data/mobilenet_ssd_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29.pb", "data/mobilenet_ssd_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29.pbtxt", cv::Size(300, 300));
    std::cout << std::endl;
}

void bench_opencv_face_detector()
{
    std::cout << "OpenCV Face Detector\n";
    bench_network("data/opencv_face_detector/deploy.prototxt", "data/opencv_face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel", cv::Size(300, 300));
    std::cout << std::endl;
}

void bench_openface_nn4_small2_v1()
{
    std::cout << "OpenFace nn4 small2 v1\n";
    bench_network("data/openface_nn4_small2_v1/nn4.small2.v1.t7", "", cv::Size(96, 96));
    std::cout << std::endl;
}

void bench_openpose_pose_mpi()
{
    std::cout << "OpenPose pose MPI\n";
    bench_network("data/openpose_pose_mpi/openpose_pose_mpi_faster_4_stages.prototxt", "data/openpose_pose_mpi/pose_iter_160000.caffemodel", cv::Size(368, 368));
    std::cout << std::endl;
}

void bench_resnet50()
{
    std::cout << "ResNet 50\n";
    bench_network("data/resnet50/ResNet-50-deploy.prototxt", "data/resnet50/ResNet-50-model.caffemodel", cv::Size(224, 224));
    std::cout << std::endl;
}

void bench_resnet50_faster_rcnn()
{
    std::cout << "ResNet50 Faster RCNN\n";
    bench_network("data/resnet50_faster_rcnn/faster_rcnn_resnet50_coco_2018_01_28.pbtxt", "data/resnet50_faster_rcnn/faster_rcnn_resnet50_coco_2018_01_28.pb", cv::Size(224, 224));
    std::cout << std::endl;
}

void bench_resnet101()
{
    std::cout << "ResNet 101\n";
    bench_network("data/resnet101/ResNet-101-deploy.prototxt", "data/resnet101/ResNet-101-model.caffemodel", cv::Size(224, 224));
    std::cout << std::endl;
}

void bench_squeezenet()
{
    std::cout << "SqueezeNet v1.1\n";
    bench_network("data/squeezenet/squeezenet_v1.1.prototxt", "data/squeezenet/squeezenet_v1.1.caffemodel", cv::Size(227, 227));
    std::cout << std::endl;
}

void bench_inception_v2_coco()
{
    std::cout << "Inception v2 Coco\n";
    bench_network("data/ssd_inception_v2_coco_2017_11_17/ssd_inception_v2_coco_2017_11_17.pb", "data/ssd_inception_v2_coco_2017_11_17/ssd_inception_v2_coco_2017_11_17.pbtxt", cv::Size(300, 300));
    std::cout << std::endl;
}

void bench_tensorflow_inception_5h()
{
    std::cout << "TensorFlow Inception 5h\n";
    bench_network("data/tensorflow_inception_5h/tensorflow_inception_graph.pb", "", cv::Size(224, 224));
    std::cout << std::endl;
}

void bench_vgg16()
{
    std::cout << "VGG16 SSD\n";
    bench_network("data/vgg16/ssd_vgg16.prototxt", "data/vgg16/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel", cv::Size(300, 300));
    std::cout << std::endl;
}

void bench_vgg16_faster_rcnn()
{
    std::cout << "VGG16 Faster RCNN\n";
    bench_network("data/vgg16_faster_rcnn/faster_rcnn_vgg16.prototxt", "data/vgg16_faster_rcnn/VGG16_faster_rcnn_final.caffemodel", cv::Size(224, 224));
    std::cout << std::endl;
}

void bench_vgg_ssd512()
{
    std::cout << "VGG SSD512\n";
    bench_network("data/vgg512/deploy.prototxt", "data/vgg512/VGG_coco_SSD_512x512_iter_360000.caffemodel", cv::Size(512, 512));
    std::cout << std::endl;
}

void bench_yolo_v2()
{
    std::cout << "YOLO v2\n";
    bench_network("data/yolov2/yolov2.cfg", "data/yolov2/yolov2.weights", cv::Size(608, 608));
    std::cout << std::endl;
}

void bench_yolo_v3()
{
    std::cout << "YOLO v3\n";
    bench_network("data/yolov3/yolov3.cfg", "data/yolov3/yolov3.weights", cv::Size(608, 608));
    std::cout << std::endl;
}

void bench_yolo_v3_spp()
{
    std::cout << "YOLO v3 SPP\n";
    bench_network("data/yolov3_spp/yolov3-spp.cfg", "data/yolov3_spp/yolov3-spp.weights", cv::Size(608, 608));
    std::cout << std::endl;
}

void bench_yolov3_enet_b0()
{
    std::cout << "EfficientNet B0 YOLOv3\n";
    bench_network("data/yolov3-enet-b0/enet-coco.cfg", "data/yolov3-enet-b0/enetb0-coco_final.weights", cv::Size(416, 416));
    std::cout << std::endl;
}

void bench_yolo_v3_tiny()
{
    std::cout << "YOLO v3 Tiny\n";
    bench_network("data/yolov3-tiny/yolov3-tiny.cfg", "data/yolov3-tiny/yolov3-tiny.weights", cv::Size(416, 416));
    std::cout << std::endl;
}

void bench_yolo_v3_tiny_prn()
{
    std::cout << "YOLO v3 Tiny PRN\n";
    bench_network("data/yolov3-tiny-prn/yolov3-tiny-prn.cfg", "data/yolov3-tiny-prn/yolov3-tiny-prn.weights", cv::Size(416, 416));
    std::cout << std::endl;
}

void bench_yolo_v4()
{
    std::cout << "YOLO v4\n";
    bench_network("data/yolov4/yolov4.cfg", "data/yolov4/yolov4.weights", cv::Size(608, 608));
    std::cout << std::endl;
}

void bench_yolo_v4_tiny()
{
    std::cout << "YOLO v4 Tiny\n";
    bench_network("data/yolov4-tiny/yolov4-tiny.cfg", "data/yolov4-tiny/yolov4-tiny.weights", cv::Size(416, 416));
    std::cout << std::endl;
}

void bench_zf_faster_rcnn()
{
    std::cout << "ZF Faster RCNN\n";
    bench_network("data/zf_faster_rcnn/faster_rcnn_zf.prototxt", "data/zf_faster_rcnn/ZF_faster_rcnn_final.caffemodel", cv::Size(224, 224));
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    constexpr auto total_images = 10;
    auto prefix = std::string("data/images/img_"),
         suffix = std::string(".jpg");

    /* populate sample images */
    for (int i = 0; i < total_images; i++) {
        auto file = prefix + std::to_string(i) + suffix;
        auto image = cv::imread(file);
        image_samples.push_back(image);
    }

    bench_yolo_v4();
    bench_yolo_v4_tiny();
    return 0;

    bench_alexnet();
    bench_densenet121();
    bench_east_text_detection();
    bench_enet();
    bench_fns_stary_night();
    bench_googlenet();
    bench_inception_v2_faster_rcnn();
    bench_inception_v2_mask_rcnn();
    bench_mobilenet_ssd();
    bench_mobilenet_ssd_v1_coco();
    bench_mobilenet_ssd_v2_coco();
    bench_opencv_face_detector();
    bench_openface_nn4_small2_v1();
    bench_openpose_pose_mpi();
    bench_resnet50();
    bench_resnet50_faster_rcnn();
    bench_resnet101();
    bench_squeezenet();
    bench_inception_v2_coco();
    bench_tensorflow_inception_5h();
    bench_vgg16();
    bench_vgg_ssd512();
    bench_vgg16_faster_rcnn();
    bench_yolo_v2();
    bench_yolo_v3_tiny();
    bench_yolo_v3_tiny_prn();
    bench_yolo_v3();
    bench_yolo_v3_spp();
    bench_yolov3_enet_b0();
    bench_yolo_v4();
    bench_yolo_v4_tiny();
    bench_zf_faster_rcnn();
    return 0;
}

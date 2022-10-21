#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "utils.hpp"
#include <malloc.h>
#include <opencv2/opencv.hpp>
#include "rapidjson/document.h"

#define USE_FP32 //USE_FP16
#define INPUT_NAME "data"
#define OUTPUT_NAME "prob"
#define MAX_BATCH_SIZE 8

using namespace nvinfer1;
using namespace std;

//Debugging messages (info, warning, error, or internal error/fatal)
static Logger gLogger;

float sigmoid(float x) {
    return (1 / (1 + exp(-x)));
}

static std::vector<BlockArgs>
        block_args_list = {
        BlockArgs{1, 3, 1, 1, 32, 16, 0.25, true},
        BlockArgs{2, 3, 2, 6, 16, 24, 0.25, true},
        BlockArgs{2, 5, 2, 6, 24, 40, 0.25, true},
        BlockArgs{3, 3, 2, 6, 40, 80, 0.25, true},
        BlockArgs{3, 5, 1, 6, 80, 112, 0.25, true},
        BlockArgs{4, 5, 2, 6, 112, 192, 0.25, true},
        BlockArgs{1, 3, 1, 6, 192, 320, 0.25, true}};

static std::map<std::string, GlobalParams>
        global_params_map = {
        // input_h,input_w,num_classes,batch_norm_epsilon,
        // width_coefficient,depth_coefficient,depth_divisor, min_depth
        {"b0",        GlobalParams{224, 224, 1000, 0.001, 1.0, 1.0, 8, -1}},
        {"b1",        GlobalParams{240, 240, 1000, 0.001, 1.0, 1.1, 8, -1}},
        {"b2",        GlobalParams{260, 260, 1000, 0.001, 1.1, 1.2, 8, -1}},
        {"b3",        GlobalParams{300, 300, 1000, 0.001, 1.2, 1.4, 8, -1}},
        {"b4",        GlobalParams{380, 380, 1000, 0.001, 1.4, 1.8, 8, -1}},
        {"b5",        GlobalParams{456, 456, 1000, 0.001, 1.6, 2.2, 8, -1}},
        {"b6",        GlobalParams{528, 528, 1000, 0.001, 1.8, 2.6, 8, -1}},
        {"b7",        GlobalParams{600, 600, 1000, 0.001, 2.0, 3.1, 8, -1}},
        {"b8",        GlobalParams{672, 672, 1000, 0.001, 2.2, 3.6, 8, -1}},
        {"l2",        GlobalParams{800, 800, 1000, 0.001, 4.3, 5.3, 8, -1}},
        {"b3_flower", GlobalParams{300, 300, 17, 0.001, 1.2, 1.4, 8, -1}},
};

vector<float> prepareImage(const cv::Mat &img, const GlobalParams &global_params) {
    int c = 3;
    int h = global_params.input_h;
    int w = global_params.input_w;

    auto scaleSize = cv::Size(w, h);

    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, scaleSize, 0, 0, cv::INTER_CUBIC);

    cv::Mat img_float;
    resized.convertTo(img_float, CV_32FC3, 1.f / 255.0);

    //HWC TO CHW
    vector<cv::Mat> input_channels(c);
    cv::split(img_float, input_channels);

    vector<float> result(h * w * c);
    auto data = result.data();
    int channel_length = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data, input_channels[i].data, channel_length * sizeof(float));
        data += channel_length;  // 指针后移channel_length个单位
    }
    return result;
}

bool readFileToString(string file_name, string &fileData) {

    ifstream file(file_name.c_str(), std::ifstream::binary);

    if (file) {
        // Calculate the file's size, and allocate a buffer of that size.
        file.seekg(0, file.end);
        const int file_size = file.tellg();
        char *file_buf = new char[file_size + 1];
        //make sure the end tag \0 of string.
        memset(file_buf, 0, file_size + 1);

        // Read the entire file into the buffer.
        file.seekg(0, ios::beg);
        file.read(file_buf, file_size);

        if (file) {
            fileData.append(file_buf);
        } else {
            std::cout << "error: only " << file.gcount() << " could be read";
            fileData.append(file_buf);
            return false;
        }
        file.close();
        delete[]file_buf;
    } else {
        return false;
    }

    return true;
}

ICudaEngine *
createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt, std::string path_wts,
             std::vector<BlockArgs> block_args_list, GlobalParams global_params) {
    cout << "create Engine run" << endl;
    float bn_eps = global_params.batch_norm_epsilon;
    DimsHW image_size = DimsHW{global_params.input_h, global_params.input_w};

    std::map<std::string, Weights> weightMap = loadWeights(path_wts);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    INetworkDefinition *network = builder->createNetworkV2(0U);
    ITensor *data = network->addInput(INPUT_NAME, dt, Dims3{3, global_params.input_h, global_params.input_w});
    assert(data);

    int out_channels = roundFilters(32, global_params);
    auto conv_stem = addSamePaddingConv2d(network, weightMap, *data, out_channels, 3, 2, 1, 1, image_size,
                                          "_conv_stem");
    auto bn0 = addBatchNorm2d(network, weightMap, *conv_stem->getOutput(0), "_bn0", bn_eps);
    auto swish0 = addSwish(network, *bn0->getOutput(0));
    ITensor *x = swish0->getOutput(0);
    image_size = calculateOutputImageSize(image_size, 2);
    int block_id = 0;
    for (int i = 0; i < block_args_list.size(); i++) {
        BlockArgs block_args = block_args_list[i];

        block_args.input_filters = roundFilters(block_args.input_filters, global_params);
        block_args.output_filters = roundFilters(block_args.output_filters, global_params);
        block_args.num_repeat = roundRepeats(block_args.num_repeat, global_params);
        x = MBConvBlock(network, weightMap, *x, "_blocks." + std::to_string(block_id), block_args, global_params,
                        image_size);

        assert(x);
        block_id++;
        image_size = calculateOutputImageSize(image_size, block_args.stride);
        if (block_args.num_repeat > 1) {
            block_args.input_filters = block_args.output_filters;
            block_args.stride = 1;
        }
        for (int r = 0; r < block_args.num_repeat - 1; r++) {
            x = MBConvBlock(network, weightMap, *x, "_blocks." + std::to_string(block_id), block_args, global_params,
                            image_size);
            block_id++;
        }
    }
    out_channels = roundFilters(1280, global_params);
    auto conv_head = addSamePaddingConv2d(network, weightMap, *x, out_channels, 1, 1, 1, 1, image_size, "_conv_head",
                                          false);
    auto bn1 = addBatchNorm2d(network, weightMap, *conv_head->getOutput(0), "_bn1", bn_eps);
    auto swish1 = addSwish(network, *bn1->getOutput(0));
    auto avg_pool = network->addPoolingNd(*swish1->getOutput(0), PoolingType::kAVERAGE, image_size);

    IFullyConnectedLayer *final = network->addFullyConnected(*avg_pool->getOutput(0), global_params.num_classes,
                                                             weightMap["_fc.weight"], weightMap["_fc.bias"]);
    assert(final);

    final->getOutput(0)->setName(OUTPUT_NAME);
    network->markOutput(*final->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 24);
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "build engine ..." << std::endl;

    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine != nullptr);

    std::cout << "build finished" << std::endl;
    // Don't need the network anymore
    network->destroy();
    // Release host memory
    for (auto &mem: weightMap) {
        free((void *) (mem.second.values));
    }

    return engine;
}

///利用API创建模型
void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream, std::string wtsPath,
                std::vector<BlockArgs> block_args_list, GlobalParams global_params) {
    // Create builder
    IBuilder *builder = createInferBuilder(gLogger);
    cout << "create Infer Builder finished" << std::endl;

    //创建engine的详细信息
    IBuilderConfig *config = builder->createBuilderConfig();
    cout << "create Builder Config finished" << std::endl;

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT, wtsPath, block_args_list,
                                       global_params);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext &context, float *input, float *output, int batchSize, GlobalParams global_params) {
    const ICudaEngine &engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex],
                     batchSize * 3 * global_params.input_h * global_params.input_w * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * global_params.num_classes * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input,
                          batchSize * 3 * global_params.input_h * global_params.input_w * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * global_params.num_classes * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

bool parse_args(int argc, char **argv, std::string &wts, std::string &engine, std::string &backbone) {
    if (std::string(argv[1]) == "-s" && argc == 5) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        backbone = std::string(argv[4]);
    } else if (std::string(argv[1]) == "-d" && argc == 4) {
        engine = std::string(argv[2]);
        backbone = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    std::string wtsPath = "/home/zjd/clion_workspace/efficientnet_b3_flowers.wts";
    std::string engine_name = "efficientnet_b3_flowers.engine";
    std::string backbone = "b3_flower";

    //INFO of Net struct
    GlobalParams global_params = global_params_map[backbone];
    std::ifstream file(engine_name, std::ios::binary);

    ///wts-->engine ***************************
    // create a model using the API directly and serialize it to a stream
    if (!file.good()) {   // Engine是否已经生成
        if (!wtsPath.empty()) {
            //利用IHostMemory创建一个modelStream用于后面API写入engine
            IHostMemory *modelStream{nullptr};
            cout << "API To Model run" << endl;
            APIToModel(MAX_BATCH_SIZE, &modelStream, wtsPath, block_args_list, global_params);
            cout << "API To Model finished" << endl;
            assert(modelStream != nullptr);

            std::ofstream p(engine_name, std::ios::binary);
            if (!p) {
                std::cerr << "could not open plan output file" << std::endl;
                return -1;
            }
            //reinterpret_cast-数据类型转换
            p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
            modelStream->destroy();
            return 1;
        }
    }
    ///*****************************************

    char *trtModelStream{nullptr};
    size_t size{0};
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    } else {
        std::cerr << "could not open plan file" << std::endl;
        return -1;
    }


    ///image input
    float *data = new float[3 * global_params.input_h * global_params.input_w];
    cout << "input_h=" << global_params.input_h << " input_w=" << global_params.input_w << endl;
    cout << "data input len=" << (malloc_usable_size(data) / sizeof(*data)) << endl;
    vector<float> image_data;
    cv::Mat img = cv::imread("/home/zjd/clion_workspace/OxFlower17/test/17/1291.jpg", 1);
    image_data = prepareImage(img,global_params);
    cout << "image_data=" << image_data.size() << endl;
    for (int k = 0; k < image_data.size(); k++) {
        data[k] = image_data[k];
    }


    ///engine-->inference **********************
    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    cout<<"Run inference 100 case:"<<endl;
    float *prob = new float[global_params.num_classes];
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1, global_params);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
    ///******************************************


    ///inference Output
    float max_confidence = -9999;
    int max_index = 0;
    int len_of_prob = (malloc_usable_size(prob) / sizeof(*prob));
    cout << "output len=" << (malloc_usable_size(prob) / sizeof(*prob)) << endl;
    for (int k = 0; k < len_of_prob; k++) {
        if (prob[k] > max_confidence) {
            max_confidence = sigmoid(prob[k]);
            max_index = k;
        }
    }
    cout << "max_confidence=" << max_confidence << endl;
    cout << "max_index=" << max_index << endl;

    string json_fileName("/home/zjd/tensorrtx/efficientnet/flowers_labels.txt");
    string label_class_str;
    readFileToString(json_fileName, label_class_str);

    const char *json1 = label_class_str.c_str();
    rapidjson::Document document1;
    document1.Parse(json1);
    string class_index = to_string(max_index);
    cout << "Class: " << document1[class_index.c_str()].GetString() << endl;

    for (unsigned int i = 0; i < len_of_prob; i++) {
        std::cout << sigmoid(prob[i]) << ", ";
    }
    std::cout << std::endl;


    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    delete data;
    delete prob;

    return 0;
}

# EfficientNet-TensorRT

#### Environment：Ubuntu20.04 、CUDA>11.0 、cudnn>8.2 、cmake >3.1  、TensorRT>8.0

TensorRT building model reasoning generally has three ways: 

- (1) use the TensorRT interface provided by the framework, such as TF-TRT and Torch TRT;

- (2) Use Parser front-end interpreter, such as TF/Torch/... ->ONNX ->TensorRT;

- (3) Use TensorRT native API to build the network.
  
  Of course, the difficulty and ease of use must be from low to high, and the accompanying performance and compatibility are also from low to high. Here we will introduce the third method directly.

#### Construction phase

Logger、Builder、BuilderConfig、Network、SerializedNetwork

#### Run Time

Engine、Context、Buffer correlation、Execute

### what TensorRT have did？

TensorRT deeply optimizes the operation efficiency of reasoning

- Automatically select the optimal kernel
  
  - There are multiple CUDA implementations for matrix multiplication and convolution, and the optimal implementation is automatically selected according to the size and shape of the data

- Calculation chart optimization
  
  - Generate network optimization calculation diagram by means of kernel integration and reducing data copy

- Support fp16/int8
  
  - Precision conversion and scaling of numerical values, making full use of hardware's low precision and high throughput computing capabilities

### How to build？？

### Warning ! some path you have to change:

CMakeLists.txt

```cmake
# tensorrt
include_directories(xxx/TensorRT-8.4.3.1/include/)
link_directories(xxx/TensorRT-8.4.3.1/lib/)
```

efficientnet.cpp

```cpp
///line40:define your model global params
static std::map<std::string, GlobalParams>
        global_params_map = {
        // input_h,input_w,num_classes,batch_norm_epsilon,
        // width_coefficient,depth_coefficient,depth_divisor, min_depth
        {"b3_flower", GlobalParams{300, 300, 17, 0.001, 1.2, 1.4, 8, -1}},
}; //add your own efficientnet train params


///line274: your onw model wts and name
std::string wtsPath = "/home/zjd/clion_workspace/efficientnet_b3_flowers.wts";
std::string engine_name = "efficientnet_b3_flowers.engine";
std::string backbone = "b3_flower";

///line327: input image
cv::Mat img = cv::imread("/home/zjd/clion_workspace/OxFlower17/test/17/1291.jpg", 1);


///line370: maybe a json file to save the class labels 
string json_fileName("/home/zjd/tensorrtx/efficientnet/flowers_labels.txt");

```

Then make build ! ! !

```shell
cd ./EfficientNet-TensorRT
mkdir build
cd build
cmake ..
make
```

```shell
./efficientnet #filrst build engine
[output]:build finisd

./efficientnet #second run inference
[output]:Class label:xxxx
```

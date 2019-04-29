//
#include <stdlib.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <cuda.h>
#include <cudnn.h>

// Get compiler version
#define DEFTOSTR_(s)   #s
#define DEFTOSTR(s)    DEFTOSTR_(s)
#if defined(__GNUC__)
#define COMPILER_NAME "GCC"
#define COMPILER_VER  DEFTOSTR(__GNUC__) "." DEFTOSTR(__GNUC_MINOR__) "." DEFTOSTR(__GNUC_PATCHLEVEL__)
#elif defined(_MSC_VER)
#if _MSC_VER < 1200
#define COMPILER_NAME "MSVC 05"
#elif _MSC_VER < 1300
#define COMPILER_NAME "MSVC 06"
#elif _MSC_VER < 1400
#define COMPILER_NAME "MSVC 07"
#elif _MSC_VER < 1500
#define COMPILER_NAME "MSVC 08"
#elif _MSC_VER < 1600
#define COMPILER_NAME "MSVC 09"
#elif _MSC_VER < 1700
#define COMPILER_NAME "MSVC 10"
#elif _MSC_VER < 1800
#define COMPILER_NAME "MSVC 11"
#elif _MSC_VER < 1900
#define COMPILER_NAME "MSVC 12"
#elif _MSC_VER < 2000
#define COMPILER_NAME "MSVC 14"
#else
#define COMPILER_NAME "MSVC"
#endif
#define COMPILER_VER  DEFTOSTR(_MSC_FULL_VER) "." DEFTOSTR(_MSC_BUILD)
#elif defined(__clang_major__)
#define COMPILER_NAME "CLANG"
#define COMPILER_VER  DEFTOSTR(__clang_major__ ) "." DEFTOSTR(__clang_minor__) "." DEFTOSTR(__clang_patchlevel__)
#elif defined(__INTEL_COMPILER)
#define COMPILER_NAME "ICC"
#define COMPILER_VER DEFTOSTR(__INTEL_COMPILER) "." DEFTOSTR(__INTEL_COMPILER_BUILD_DATE)
#else
#define COMPILER_NAME "unknown"
#define COMPILER_VER  "???"
#endif

// Get cuDNN version
#define CUDNN_VERSION_STR  DEFTOSTR(CUDNN_MAJOR) "." DEFTOSTR (CUDNN_MINOR) "." DEFTOSTR(CUDNN_PATCHLEVEL)

#define FatalError(s) {                                 \
    std::stringstream _where, _message;                 \
    _where << __FILE__ << ':' << __LINE__;              \
    _message << std::string(s) + "\n" << _where.str();  \
    std::cerr << _message.str() << "\nAborting...\n";   \
    cudaDeviceReset();                                  \
    exit(EXIT_FAILURE);                                 \
}

#define checkCUDNN(expression) {                \
    cudnnStatus_t status = (expression);        \
    if (status != CUDNN_STATUS_SUCCESS) {       \
        std::stringstream _where;               \
        _where << __FILE__ << ':' << __LINE__;  \
        std::cerr << "ERROR: "                  \
            << cudnnGetErrorString(status)      \
            << "\n" << _where.str() << "\n";    \
        std::exit(EXIT_FAILURE);                \
    }                                           \
}

void checkCudaErrors(cudaError_t status) {
    std::stringstream _error;
    if (status != 0) {
      _error << "Cuda failure\nError: " << cudaGetErrorString(status);
      FatalError(_error.str());
    }
    return;
}

static void showDevices() {
    int totalDevices;
    checkCudaErrors(cudaGetDeviceCount(&totalDevices));
    printf("\nThere are %d CUDA capable devices on your machine :\n", totalDevices);
    for (int i = 0; i < totalDevices; i++) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, i));
        printf("device %d : sms %2d  Capabilities %d.%d, SmClock %.1f Mhz, MemSize (Mb) %d, MemClock %.1f Mhz, Ecc=%d, boardGroupID=%d\n",
               i, prop.multiProcessorCount, prop.major, prop.minor,
               static_cast<float>(prop.clockRate*1e-3),
               static_cast<int>(prop.totalGlobalMem/(1024*1024)),
               static_cast<float>(prop.memoryClockRate*1e-3),
               prop.ECCEnabled,
               prop.multiGpuBoardGroupID);
    }
}

void print4d_nchw (const float* buffer, int batch_size, int channels, int height, int width) {
    for (int batch = 0; batch < batch_size; ++batch) {
        std::cout << "{\n";
        for (int channel = 0; channel < channels; ++channel) {
            std::cout << "  {\n";
            for (int row = 0; row < height; ++row) {
                std::cout << "    {";
                for (int column = 0; column < width; ++column) {
                    std::cout << " ";
                    std::cout << static_cast<float>(buffer[batch*channels*height*width + channel*height*width + row*width + column]);
                    if (column + 1 != width)
                        std::cout << ",";
                }
                if (row + 1 == height)
                    std::cout << "}\n";
                else
                    std::cout << "},\n";
            }
            if (channel + 1 == channels)
                std::cout << "  }\n";
            else
                std::cout << "  },\n";
        }
        if (batch + 1 == batch_size)
            std::cout << "}\n";
        else
            std::cout << "};\n";
    }
    return;
}

void print4d_nhwc (const float* buffer, int batch_size, int channels, int height, int width) {
    for (int batch = 0; batch < batch_size; ++batch) {
        std::cout << "{\n";
        for (int row = 0; row < height; ++row) {
            std::cout << "  {\n";
            for (int column = 0; column < width; ++column) {
                std::cout << "    {";
                for (int channel = 0; channel < channels; ++channel) {
                    std::cout << " ";
                    std::cout << static_cast<float>(buffer[batch*height*width*channels + row*width*channels + column*channels + channel]);
                    if (channel + 1 != channels)
                        std::cout << ",";
                }
                if (column + 1 == width)
                    std::cout << "}\n";
                else
                    std::cout << "},\n";
            }
            if (row + 1 == height)
                std::cout << "  }\n";
            else
                std::cout << "  },\n";
        }
        if (batch + 1 == batch_size)
            std::cout << "}\n";
        else
            std::cout << "};\n";
    }
    return;
}

int main( [[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    int version = static_cast<int>(cudnnGetVersion());
    printf("cudnnGetVersion(): %d, CUDNN_VERSION from cudnn.h: %d (%s)\n", version, CUDNN_VERSION, CUDNN_VERSION_STR);
    printf("Host compiler version: %s %s\r", COMPILER_NAME, COMPILER_VER);
    showDevices();
    cudaSetDevice(0);
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    // Mystery image
    const float image[1][3][4][4] = {
        {
            {{0.25,   1.0,  0.0, 0.9}, {0.15,  0.3, 0.57, 0.26}, {0.22,  0.1, 0.12, 0.5}, {0.34, 0.99, 0.02, 0.05}},
            {{0.16,  0.14, 0.36, 0.7}, {0.86, 0.68, 0.11,  0.2}, {0.76, 0.99, 0.54, 0.9}, {0.37, 0.27, 0.45, 0.83}},
            {{0.91,  0.18, 0.66, 0.5}, {0.51, 0.33,  1.0,  0.2}, {0.34, 0.97, 0.88, 0.5}, {0.78, 0.72, 0.39, 0.65}},
        }
    };
    // for debug
    // float* bla{nullptr};
    // bla = (float*)malloc(48*sizeof(float));
    // memcpy(bla, &image, 48*sizeof(float));
    // print4d_nchw(bla,1,3,4,4);

    // Mystery kernel
    const float kernel_template[3][3] = {
        {5.2,  1.9, 7.1},
        {17.5, -8., 20.},
        {4.7,  1.4, 4.}
    };
    // input descriptor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
            input_descriptor,
            CUDNN_TENSOR_NHWC,  // format
            CUDNN_DATA_FLOAT,   // dataType
            1,  // batch_size
            3,  // channels
            4,  // image_height
            4   // image_width
        )
    );
    // kernel descriptor
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(
            kernel_descriptor,
            CUDNN_DATA_FLOAT, // dataType
            CUDNN_TENSOR_NCHW,  // format
            3,  // out_channels
            3,  // in_channels
            3,  // kernel_height
            3  // kernel_width
        )
    );
    // convolution descriptor
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
            convolution_descriptor,
            1,  // pad_height
            1,  // pad_width
            1,  // vertical_stride
            1,  // horizonal_stride
            1,  // dilation_height
            1,  // dilation_width
            CUDNN_CROSS_CORRELATION,  // mode
            CUDNN_DATA_FLOAT  // computeType
        )
    );
    // output dims
    int batch_size, channels, height, width;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
            convolution_descriptor,  // convolution descriptor
            input_descriptor,  // input descriptor
            kernel_descriptor,  // kernel desciptor
            &batch_size,  // batch_size
            &channels,  // channels
            &height,  // height
            &width  // width
        )
    );
    std::cout << "Dims: " << batch_size << " " << channels << " " << height << " " << width << "\n";
    // output descriptor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
            output_descriptor,
            CUDNN_TENSOR_NHWC,  // format
            CUDNN_DATA_FLOAT,  // dataType
            batch_size,  // batch_size
            channels,  // channels
            height,  // image_height
            width  // image_width
        )
    );
    // algorithm descriptor
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
            cudnn,
            input_descriptor,
            kernel_descriptor,
            convolution_descriptor,
            output_descriptor,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,  // memoryLimitInBytes
            &convolution_algorithm
        )
    );
    size_t workspace_bytes{0};
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn,
            input_descriptor,
            kernel_descriptor,
            convolution_descriptor,
            output_descriptor,
            convolution_algorithm,
            &workspace_bytes
        )
    );
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
        << std::endl;

    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    int image_bytes = 1 * 3 * 4 * 4 * sizeof(float);  // batch_size * channels * height * width of input
    float* d_input{nullptr};
    cudaMalloc(&d_input, image_bytes);
    cudaMemcpy(d_input, &image, image_bytes, cudaMemcpyHostToDevice);

    int output_bytes = batch_size * channels * height * width * sizeof(float);  // batch_size * channels * height * width of output
    float* d_output{nullptr};
    cudaMalloc(&d_output, output_bytes);
    // cudaMemset(d_output, 0, output_bytes); // not needed for 0s

    float h_kernel[3][3][3][3];
    for (int kernel = 0; kernel < 3; ++kernel) {
        for (int channel = 0; channel < 3; ++channel) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    h_kernel[kernel][channel][row][column] = kernel_template[row][column];
                }
            }
        }
    }
    // for debug
    // float* h_kernel_check{nullptr};
    // h_kernel_check = (float*)malloc(81*sizeof(float));
    // memcpy(h_kernel_check, &h_kernel, 81*sizeof(float));
    // print4d_nchw(h_kernel_check, 3, 3, 3, 3);

    float* d_kernel{nullptr};
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    // convolve
    const float alpha = 1, beta = 0;
    checkCUDNN(cudnnConvolutionForward(
            cudnn,
            &alpha,
            input_descriptor,
            d_input,
            kernel_descriptor,
            d_kernel,
            convolution_descriptor,
            convolution_algorithm,
            d_workspace,
            workspace_bytes,
            &beta,
            output_descriptor,
            d_output
        )
    );

    float* h_output = new float[output_bytes];
    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    // Print output for debug
    // print4d_nhwc(h_output, batch_size, channels, height, width);

    // check output
    const bool check = std::abs(h_output[0] - 20.29) < 0.01;

    delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);
    if (!check) {
        std::cout << "Failure...\n";
        exit(EXIT_FAILURE);
    }
    std::cout << "Success!\n";
    return EXIT_SUCCESS;
}

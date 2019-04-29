### Some very simple tests for Arch Linux packages

```diff
- DISCLAIMER: This is in no way or form official and should not be viewed as representing Arch Linux
```
#### Currently supported
 * pytorch
   - C++ API
   - Python API
 * tensorflow
   - C++ API
   - Python API
   - Keras API
 * cuDNN
   - C++ API

#### Running the tests
```bash
make
```
#### Example Output
```bash
$ make  # on CPU
make -C pytorch/ATan run
make[1]: Entering directory '/home/gizdov/Tests/arch-package-tests/pytorch/ATan'
./atan
CUDA device is not available.
 1  0
 1  1
[ CPUIntType{2,2} ]
make[1]: Leaving directory '/home/gizdov/Tests/arch-package-tests/pytorch/ATan'
make -C pytorch/autograd run
make[1]: Entering directory '/home/gizdov/Tests/arch-package-tests/pytorch/autograd'
./autograd
CUDA device is not available.
We are using CPU
 1.0782  1.3101
 1.5989  1.8083
[ Variable[CPUFloatType]{2,2} ]
make[1]: Leaving directory '/home/gizdov/Tests/arch-package-tests/pytorch/autograd'
make -C tensorflow/basic run
make[1]: Entering directory '/home/gizdov/Tests/arch-package-tests/tensorflow/basic'
./basic
2019-04-16 15:20:08.215189: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2593000000 Hz
2019-04-16 15:20:08.215431: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x55fac13fa840 executing computations on platform Host. Devices:
2019-04-16 15:20:08.215451: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2019-04-16 15:20:08.217658: I basic.cpp:22] 19
-3
make[1]: Leaving directory '/home/gizdov/Tests/arch-package-tests/tensorflow/basic'
#########################################
############### COMPLETED ###############
#########################################
```

```bash
$ make  # on GPU
make -C pytorch/ATan run
make[1]: Entering directory '/home/gizdov/Git/arch-package-tests/pytorch/ATan'
g++ atan.cpp -I/usr/include/torch/csrc/api/include -I/usr/include/python3.7m -L/usr/lib/pytorch -L/opt/cuda/lib -lc10 -ltorch -lcaffe2 -lnvrtc -lcuda -o atan
./atan
CUDA device is available.
 0  1
-1  1
[ CUDAIntType{2,2} ]
make[1]: Leaving directory '/home/gizdov/Git/arch-package-tests/pytorch/ATan'
make -C pytorch/autograd run
make[1]: Entering directory '/home/gizdov/Git/arch-package-tests/pytorch/autograd'
g++ autograd.cpp -I/usr/include/torch/csrc/api/include -I/usr/include/python3.7m -L/usr/lib/pytorch -L/opt/cuda/lib -lc10 -ltorch -lcaffe2 -lnvrtc -lcuda -o autograd
./autograd
CUDA device is available.
We are using CUDA
 1.5140  0.8996
 1.2108  0.9279
[ Variable[CUDAFloatType]{2,2} ]
make[1]: Leaving directory '/home/gizdov/Git/arch-package-tests/pytorch/autograd'
make -C tensorflow/basic run
make[1]: Entering directory '/home/gizdov/Git/arch-package-tests/tensorflow/basic'
g++ basic.cpp -I/usr/lib/python3.7/site-packages/tensorflow/include -ltensorflow_cc -ltensorflow_framework -o basic
./basic
2019-04-16 13:21:42.975410: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3401190000 Hz
2019-04-16 13:21:42.976343: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x559ffc6dd700 executing computations on platform Host. Devices:
2019-04-16 13:21:42.976364: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2019-04-16 13:21:43.026726: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:998] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-04-16 13:21:43.027210: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x559ffc9f4160 executing computations on platform CUDA. Devices:
2019-04-16 13:21:43.027225: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): GeForce GTX 980, Compute Capability 5.2
2019-04-16 13:21:43.027947: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 980 major: 5 minor: 2 memoryClockRate(GHz): 1.329
pciBusID: 0000:01:00.0
totalMemory: 3.95GiB freeMemory: 3.69GiB
2019-04-16 13:21:43.027961: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-04-16 13:21:44.977808: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-04-16 13:21:44.977843: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-04-16 13:21:44.977851: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-04-16 13:21:44.978213: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3406 MB memory) -> physical GPU (device: 0, name: GeForce GTX 980, pci bus id: 0000:01:00.0, compute capability: 5.2)
2019-04-16 13:21:45.040475: I basic.cpp:22] 19
-3
make[1]: Leaving directory '/home/gizdov/Git/arch-package-tests/tensorflow/basic'
#########################################
############### COMPLETED ###############
#########################################
```

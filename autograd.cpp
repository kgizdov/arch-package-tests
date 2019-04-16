#include <iostream>
#include <torch/extension.h>

int main() {
  if (torch::cuda::is_available()) {
    std::cout << "CUDA device is available.\n";
  } else {
    std::cout << "CUDA device is not available.\n";
  }
  try {
    auto test = torch::ones({2, 2}, at::requires_grad()).cuda();
    if (test.device().type() == torch::kCUDA) {
      std::cout << "We are using CUDA\n";
    } else if (test.device().type() == torch::kCPU) {
      std::cout << "Oops... we are supposed to run on CUDA GPU, but we are running on CPU.";
    } else {
      std::cout << "Uuuhm, we are supposed to run on CUDA GPU, but we are not even running on CPU.";
    }
    auto a = torch::ones({2, 2}, at::requires_grad()).cuda();
    auto b = torch::randn({2, 2}).cuda();
    auto c = a + b;
    c.backward(); // a.grad() will now hold the gradient of c w.r.t. a.
    std::cout << c << "\n";
  } catch (const c10::Error& e) {
    auto test = torch::ones({2, 2}, at::requires_grad());
    if (test.device().type() == torch::kCPU) {
      std::cout << "We are using CPU\n";
    } else if (test.device().type() == torch::kCUDA) {
      std::cout << "Oops... we are supposed to run on CPU, but we are running on CUDA GPU.";
    } else {
      std::cout << "Uuuhm, we are supposed to run on CPU, but we are not even running on CUDA GPU.";
    }
    auto a = torch::ones({2, 2}, at::requires_grad());
    auto b = torch::randn({2, 2});
    auto c = a + b;
    c.backward(); // a.grad() will now hold the gradient of c w.r.t. a.
    std::cout << c << "\n";
  }
  return 0;
}

#include <iostream>
#include <torch/extension.h>

int main() {
  if (torch::cuda::is_available()) {
    std::cout << "CUDA device is available.\n";
  } else {
    std::cout << "CUDA device is not available.\n";
  }
  try {
    auto test = torch::ones({2, 2}, torch::TensorOptions().requires_grad(true).device(torch::kCUDA));
    if (test.device().type() == torch::kCUDA) {
      std::cout << "We are using CUDA\n";
    } else if (test.device().type() == torch::kCPU) {
      std::cout << "Oops... we are supposed to run on CUDA GPU, but we are running on CPU.";
    } else {
      std::cout << "Uuuhm, we are supposed to run on CUDA GPU, but we are not even running on CPU.";
    }
    auto a = torch::ones({2, 2}, torch::TensorOptions().dtype(torch::kFloat).requires_grad(true).device(torch::kCUDA));
    std::cout << "a: " << a << "\n";
    auto b = torch::randn({2, 2}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    std::cout << "b: " << b << "\n";
    auto c = (a + b).sum();
    std::cout << "(a + b).sum(): " << c << "\n";
    c.backward();  // a.grad() will now hold the gradient of c w.r.t. a.
    std::cout << "a.grad(): " << a.grad() << "\n";
    torch::Tensor g_loss = torch::binary_cross_entropy(a, b);
    g_loss.backward();
    std::cout << "torch::binary_cross_entropy(a, b): " << g_loss << "\n";
    std::cout << "a.grad(): " << a.grad() << "\n";
  } catch (const c10::Error& e) {
    std::cout << e.what() << "\n";
    auto test = torch::ones({2, 2}, torch::TensorOptions().requires_grad(true).device(torch::kCPU));
    if (test.device().type() == torch::kCPU) {
      std::cout << "We are using CPU\n";
    } else if (test.device().type() == torch::kCUDA) {
      std::cout << "Oops... we are supposed to run on CPU, but we are running on CUDA GPU.";
    } else {
      std::cout << "Uuuhm, we are supposed to run on CPU, but we are not even running on CUDA GPU.";
    }
    auto a = torch::ones({2, 2}, torch::TensorOptions().requires_grad(true).device(torch::kCPU));
    std::cout << "a: " << a << "\n";
    auto b = torch::randn({2, 2}, torch::TensorOptions().device(torch::kCPU));
    std::cout << "b: " << b << "\n";
    auto c = (a + b).sum();
    std::cout << "(a + b).sum(): " << c << "\n";
    c.backward();  // a.grad() will now hold the gradient of c w.r.t. a.
    std::cout << "a.grad(): " << a.grad() << "\n";
    torch::Tensor g_loss = torch::binary_cross_entropy(a, b);
    g_loss.backward();
    std::cout << "torch::binary_cross_entropy(a, b): " << g_loss << "\n";
    std::cout << "a.grad(): " << a.grad() << "\n";
  }
  return 0;
}

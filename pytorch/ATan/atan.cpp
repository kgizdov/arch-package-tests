#include <iostream>
#include <torch/extension.h>

int main() {
  if (torch::cuda::is_available()) {
    std::cout << "CUDA device is available.\n";
  } else {
    std::cout << "CUDA device is not available.\n";
  }
  try {
    at::Tensor a = at::ones({2, 2}, at::kInt).cuda();
    at::Tensor b = at::randn({2, 2}).cuda();
    auto c = a + b.to(at::kInt).cuda();
    std::cout << c << "\n";
  } catch (const c10::Error& e) {
    at::Tensor a = at::ones({2, 2}, at::kInt);
    at::Tensor b = at::randn({2, 2});
    auto c = a + b.to(at::kInt);
    std::cout << c << "\n";
  }
  return 0;
}

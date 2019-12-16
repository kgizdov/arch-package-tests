// tensorflow/cc/example/example.cc

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main() {
  namespace tf = tensorflow;
  namespace tfo = tensorflow::ops;
  tf::Scope root = tf::Scope::NewRootScope();
  // Matrix A = [3 2; -1 0]
  auto A = tfo::Const(root, { {3.f, 2.f}, {-1.f, 0.f} });
  // Vector b = [3 5]
  auto b = tfo::Const(root, { {3.f, 5.f} });
  // v = Ab^T
  auto v = tfo::MatMul(root.WithOpName("v"), A, b, tfo::MatMul::TransposeB(true));
  std::vector<tf::Tensor> outputs;
  tf::ClientSession session(root);
  // Run and fetch v
  TF_CHECK_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]
  LOG(INFO) << outputs[0].matrix<float>();
  return 0;
}

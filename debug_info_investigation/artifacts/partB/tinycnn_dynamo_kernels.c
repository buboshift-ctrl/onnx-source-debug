#include <stddef.h>

/* stand-in for EmitC-generated kernels; one function per ONNX node. */

static void node_node_max_pool2d(float *buf, size_t n) {   /* maxpool stand-in, py-source: x = self.pool(x) */
/* onnx_node_name: node_max_pool2d   op=MaxPool   module_scope=pool.max_pool2d */
  for (size_t i = 0; i < n; ++i)
    buf[i] = buf[i] * 1.0f + 0.0f;   /* MaxPool */
/* onnx_node_end */
}

static void node_node_view(float *buf, size_t n) {   /* reshape stand-in, py-source: x = x.flatten(1) */
/* onnx_node_name: node_view   op=Reshape   module_scope=view */
  for (size_t i = 0; i < n; ++i)
    buf[i] = buf[i] * 1.0f + 0.0f;   /* Reshape */
/* onnx_node_end */
}

static void node_node_linear(float *buf, size_t n) {   /* gemm stand-in, py-source: x = self.relu(self.fc1(x)) */
/* onnx_node_name: node_linear   op=Gemm   module_scope=fc1.linear */
  for (size_t i = 0; i < n; ++i)
    buf[i] = buf[i] * 1.0f + 0.0f;   /* Gemm */
/* onnx_node_end */
}

static void node_node_relu(float *buf, size_t n) {   /* relu stand-in, py-source: x = self.relu(self.fc1(x)) */
/* onnx_node_name: node_relu   op=Relu   module_scope=relu.relu */
  for (size_t i = 0; i < n; ++i)
    buf[i] = buf[i] * 1.0f + 0.0f;   /* Relu */
/* onnx_node_end */
}

static void node_node_linear_1(float *buf, size_t n) {   /* gemm stand-in, py-source: return self.fc2(x) */
/* onnx_node_name: node_linear_1   op=Gemm   module_scope=fc2.linear_1 */
  for (size_t i = 0; i < n; ++i)
    buf[i] = buf[i] * 1.0f + 0.0f;   /* Gemm */
/* onnx_node_end */
}

void main_graph(float *buf, size_t n) {
  node_node_max_pool2d(buf, n);
  node_node_view(buf, n);
  node_node_linear(buf, n);
  node_node_relu(buf, n);
  node_node_linear_1(buf, n);
}

int main(void) {
  float buf[8] = {0};
  main_graph(buf, 8);
  return (int)buf[0];
}


module @jit_two_mm attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>) -> (tensor<512x512xf32> {jax.result_info = "result"}) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<512x512xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%1 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %3 = tensor.empty() : tensor<512x512xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %5 = linalg.matmul ins(%arg0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%4 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %6 = tensor.empty() : tensor<512x512xf32>
    %7 = linalg.add ins(%2, %5 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%6 : tensor<512x512xf32>) -> tensor<512x512xf32>
    return %7 : tensor<512x512xf32>
  }
}


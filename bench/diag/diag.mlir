#map2 = affine_map<(d0) -> (d0)>
module @jit_diagmat_left_mul attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1028x1028xf32>, %arg1: tensor<1028x1028xf32>) -> (tensor<1028x1028xf32> {jax.result_info = "result"}) {
    %0 = tensor.empty() : tensor<1028x1028xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1028x1028xf32>) -> tensor<1028x1028xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1028x1028xf32>, tensor<1028x1028xf32>) outs(%1 : tensor<1028x1028xf32>) -> tensor<1028x1028xf32>
    return %2 : tensor<1028x1028xf32>
  }

  func.func private @diagonal(%arg0: tensor<1028x1028xf32>) -> tensor<1028xf32> {
    %c0 = arith.constant 0.0 : f32
    %result = tensor.empty() : tensor<1028xf32>
    %init = linalg.fill ins(%c0 : f32) outs(%result : tensor<1028xf32>) -> tensor<1028xf32>
    
    %final = linalg.generic {
      indexing_maps = [#map2], 
      iterator_types = ["parallel"]
    } outs(%init : tensor<1028xf32>) {
    ^bb0(%out: f32):
      %i = linalg.index 0 : index
      %val = tensor.extract %arg0[%i, %i] : tensor<1028x1028xf32>
      linalg.yield %val : f32
    } -> tensor<1028xf32>

    return %final : tensor<1028xf32>
  }
}

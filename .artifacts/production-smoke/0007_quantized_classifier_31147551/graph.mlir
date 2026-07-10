module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "314833e397548364385e5a24c1faf5ebcd4eadc3a0d750a0bed444e2c855c4a1",
  lattice.input_names = ["x_coords", "x_features", "x_active"],
  lattice.input_roles = ["sparse_coords", "sparse_features", "sparse_active"],
  lattice.output_names = ["output"],
  lattice.output_roles = ["tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(
    %x_coords: tensor<?x4xi32>,
    %x_features: tensor<?x2xf32>,
    %x_active: tensor<1xi32>
  ) -> tensor<?x2xf32> {
    %x = lattice.sparse.make %x_coords, %x_features, %x_active {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>} : (tensor<?x4xi32>, tensor<?x2xf32>, tensor<1xi32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %stages_0_weight = lattice.weight @stages_0_weight {storage_key = "stages.0.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>, packing = #lattice.packing<int8, group_size = 32, scale_dtype = f16, mode = affine>} : !lattice.weight<conv3d, f32>
    %stages_0_bias = lattice.weight @stages_0_bias {storage_key = "stages.0.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %stages_0 = lattice.subm_conv3d %x, %stages_0_weight, %stages_0_bias {kernel_size = array<i64: 3, 3, 3>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose, %stages_1_features_in, %stages_1_active = lattice.sparse.decompose %stages_0 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x6xf32>, tensor<1xi32>)
    %stages_1_features = lattice.activation %stages_1_features_in {kind = #lattice.activation<sigmoid>, approximate = #lattice.gelu_approx<none>, alpha = 0.01 : f32, beta = 1.0 : f32, threshold = 20.0 : f32} : (tensor<?x6xf32>) -> tensor<?x6xf32>
    %stages_1 = lattice.sparse.with_features %stages_0, %stages_1_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x6xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %stages_2_weight = lattice.weight @stages_2_weight {storage_key = "stages.2.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>, packing = #lattice.packing<int8, group_size = 32, scale_dtype = f16, mode = affine>} : !lattice.weight<conv3d, f32>
    %stages_2_bias = lattice.weight @stages_2_bias {storage_key = "stages.2.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %stages_2 = lattice.subm_conv3d %stages_1, %stages_2_weight, %stages_2_bias {kernel_size = array<i64: 3, 3, 3>, dilation = array<i64: 2, 2, 2>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose1, %stages_3_features_in, %stages_3_active = lattice.sparse.decompose %stages_2 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x6xf32>, tensor<1xi32>)
    %stages_3_features = lattice.activation %stages_3_features_in {kind = #lattice.activation<sigmoid>, approximate = #lattice.gelu_approx<none>, alpha = 0.01 : f32, beta = 1.0 : f32, threshold = 20.0 : f32} : (tensor<?x6xf32>) -> tensor<?x6xf32>
    %stages_3 = lattice.sparse.with_features %stages_2, %stages_3_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x6xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %global_pool = lattice.global_pool %stages_3 {mode = #lattice.pool_mode<avg>, batch_size = 2} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>) -> tensor<?x6xf32>
    %head_weight = lattice.weight @head_weight {storage_key = "head.weight", layout = #lattice.weight_layout<linear_o_i>, packing = #lattice.packing<int8, group_size = 32, scale_dtype = f16, mode = affine>} : !lattice.weight<linear, f32>
    %head_bias = lattice.weight @head_bias {storage_key = "head.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %head = lattice.linear %global_pool, %head_weight, %head_bias  : (tensor<?x6xf32>, !lattice.weight<linear, f32>, !lattice.weight<bias, f32>) -> tensor<?x2xf32>
    return %head : tensor<?x2xf32>
  }
}

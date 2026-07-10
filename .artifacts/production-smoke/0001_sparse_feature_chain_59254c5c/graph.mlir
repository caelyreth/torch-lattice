module attributes {
  lattice.ir_version = 0,
  lattice.schema_digest = "314833e397548364385e5a24c1faf5ebcd4eadc3a0d750a0bed444e2c855c4a1",
  lattice.input_names = ["x_coords", "x_features", "x_active"],
  lattice.input_roles = ["sparse_coords", "sparse_features", "sparse_active"],
  lattice.output_names = ["output"],
  lattice.output_roles = ["sparse_tensor"],
  lattice.weight_file = "weights.safetensors"
} {
  func.func @forward(
    %x_coords: tensor<?x4xi32>,
    %x_features: tensor<?x2xf32>,
    %x_active: tensor<1xi32>
  ) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> {
    %x = lattice.sparse.make %x_coords, %x_features, %x_active {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>} : (tensor<?x4xi32>, tensor<?x2xf32>, tensor<1xi32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %stages_0_weight = lattice.weight @stages_0_weight {storage_key = "stages.0.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %stages_0_bias = lattice.weight @stages_0_bias {storage_key = "stages.0.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %stages_0 = lattice.subm_conv3d %x, %stages_0_weight, %stages_0_bias {kernel_size = array<i64: 1, 1, 1>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %stages_1_weight = lattice.weight @stages_1_weight {storage_key = "stages.1.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %stages_1_bias = lattice.weight @stages_1_bias {storage_key = "stages.1.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %stages_1 = lattice.subm_conv3d %stages_0, %stages_1_weight, %stages_1_bias {kernel_size = array<i64: 1, 1, 1>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose, %stages_2_features_in, %stages_2_active = lattice.sparse.decompose %stages_1 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x3xf32>, tensor<1xi32>)
    %stages_2_features = lattice.activation %stages_2_features_in {kind = #lattice.activation<gelu>, approximate = #lattice.gelu_approx<none>, alpha = 0.01 : f32, beta = 1.0 : f32, threshold = 20.0 : f32} : (tensor<?x3xf32>) -> tensor<?x3xf32>
    %stages_2 = lattice.sparse.with_features %stages_1, %stages_2_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x3xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose1, %stages_3_features_in, %stages_3_active = lattice.sparse.decompose %stages_2 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x3xf32>, tensor<1xi32>)
    %stages_3_weight = lattice.weight @stages_3_weight {storage_key = "stages.3.weight", layout = #lattice.weight_layout<channel_c>, packing = #lattice.packing<dense>} : !lattice.weight<channel, f32>
    %stages_3_bias = lattice.weight @stages_3_bias {storage_key = "stages.3.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %stages_3_features = lattice.layer_norm %stages_3_features_in, %stages_3_weight, %stages_3_bias {eps = 0.00001 : f32} : (tensor<?x3xf32>, !lattice.weight<channel, f32>, !lattice.weight<bias, f32>) -> tensor<?x3xf32>
    %stages_3 = lattice.sparse.with_features %stages_2, %stages_3_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x3xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %stages_4_weight = lattice.weight @stages_4_weight {storage_key = "stages.4.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %stages_4_bias = lattice.weight @stages_4_bias {storage_key = "stages.4.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %stages_4 = lattice.subm_conv3d %stages_3, %stages_4_weight, %stages_4_bias {kernel_size = array<i64: 1, 1, 1>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose2, %stages_5_features_in, %stages_5_active = lattice.sparse.decompose %stages_4 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x8xf32>, tensor<1xi32>)
    %stages_5_weight = lattice.weight @stages_5_weight {storage_key = "stages.5.weight", layout = #lattice.weight_layout<channel_c>, packing = #lattice.packing<dense>} : !lattice.weight<channel, f32>
    %stages_5_features = lattice.rms_norm %stages_5_features_in, %stages_5_weight {eps = 0.000001 : f32} : (tensor<?x8xf32>, !lattice.weight<channel, f32>) -> tensor<?x8xf32>
    %stages_5 = lattice.sparse.with_features %stages_4, %stages_5_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x8xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %stages_6_weight = lattice.weight @stages_6_weight {storage_key = "stages.6.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %stages_6 = lattice.subm_conv3d %stages_5, %stages_6_weight {kernel_size = array<i64: 1, 1, 1>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose3, %stages_7_features_in, %stages_7_active = lattice.sparse.decompose %stages_6 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x3xf32>, tensor<1xi32>)
    %stages_7_features = lattice.activation %stages_7_features_in {kind = #lattice.activation<tanh>, approximate = #lattice.gelu_approx<none>, alpha = 0.01 : f32, beta = 1.0 : f32, threshold = 20.0 : f32} : (tensor<?x3xf32>) -> tensor<?x3xf32>
    %stages_7 = lattice.sparse.with_features %stages_6, %stages_7_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x3xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %stages_8_weight = lattice.weight @stages_8_weight {storage_key = "stages.8.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %stages_8_bias = lattice.weight @stages_8_bias {storage_key = "stages.8.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %stages_8 = lattice.subm_conv3d %stages_7, %stages_8_weight, %stages_8_bias {kernel_size = array<i64: 1, 1, 1>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose4, %stages_9_features_in, %stages_9_active = lattice.sparse.decompose %stages_8 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x5xf32>, tensor<1xi32>)
    %stages_9_features = lattice.activation %stages_9_features_in {kind = #lattice.activation<silu>, approximate = #lattice.gelu_approx<none>, alpha = 0.01 : f32, beta = 1.0 : f32, threshold = 20.0 : f32} : (tensor<?x5xf32>) -> tensor<?x5xf32>
    %stages_9 = lattice.sparse.with_features %stages_8, %stages_9_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x5xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    return %stages_9 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
  }
}

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
    %x_features: tensor<?x4xf32>,
    %x_active: tensor<1xi32>
  ) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> {
    %x = lattice.sparse.make %x_coords, %x_features, %x_active {stride = array<i64: 1, 1, 1>, coord_order = #lattice.coord<batch_x_y_z>} : (tensor<?x4xi32>, tensor<?x4xf32>, tensor<1xi32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %left_0_weight = lattice.weight @left_0_weight {storage_key = "left.0.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %left_0 = lattice.conv3d %x, %left_0_weight {kernel_size = array<i64: 1, 1, 1>, stride = array<i64: 1, 1, 1>, padding = array<i64: 0, 0, 0>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose, %left_1_features_in, %left_1_active = lattice.sparse.decompose %left_0 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x4xf32>, tensor<1xi32>)
    %left_1_features = lattice.activation %left_1_features_in {kind = #lattice.activation<relu>, approximate = #lattice.gelu_approx<none>, alpha = 0.01 : f32, beta = 1.0 : f32, threshold = 20.0 : f32} : (tensor<?x4xf32>) -> tensor<?x4xf32>
    %left_1 = lattice.sparse.with_features %left_0, %left_1_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x4xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose1, %left_2_features_in, %left_2_active = lattice.sparse.decompose %left_1 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x4xf32>, tensor<1xi32>)
    %left_2_weight = lattice.weight @left_2_weight {storage_key = "left.2.weight", layout = #lattice.weight_layout<channel_c>, packing = #lattice.packing<dense>} : !lattice.weight<channel, f32>
    %left_2_bias = lattice.weight @left_2_bias {storage_key = "left.2.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %left_2_features = lattice.layer_norm %left_2_features_in, %left_2_weight, %left_2_bias {eps = 0.00001 : f32} : (tensor<?x4xf32>, !lattice.weight<channel, f32>, !lattice.weight<bias, f32>) -> tensor<?x4xf32>
    %left_2 = lattice.sparse.with_features %left_1, %left_2_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x4xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %right_0_weight = lattice.weight @right_0_weight {storage_key = "right.0.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %right_0_bias = lattice.weight @right_0_bias {storage_key = "right.0.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %right_0 = lattice.conv3d %x, %right_0_weight, %right_0_bias {kernel_size = array<i64: 1, 1, 1>, stride = array<i64: 1, 1, 1>, padding = array<i64: 0, 0, 0>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>, !lattice.weight<bias, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose2, %right_1_features_in, %right_1_active = lattice.sparse.decompose %right_0 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x4xf32>, tensor<1xi32>)
    %right_1_features = lattice.activation %right_1_features_in {kind = #lattice.activation<relu>, approximate = #lattice.gelu_approx<none>, alpha = 0.01 : f32, beta = 1.0 : f32, threshold = 20.0 : f32} : (tensor<?x4xf32>) -> tensor<?x4xf32>
    %right_1 = lattice.sparse.with_features %right_0, %right_1_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x4xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose3, %right_2_features_in, %right_2_active = lattice.sparse.decompose %right_1 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x4xf32>, tensor<1xi32>)
    %right_2_weight = lattice.weight @right_2_weight {storage_key = "right.2.weight", layout = #lattice.weight_layout<channel_c>, packing = #lattice.packing<dense>} : !lattice.weight<channel, f32>
    %right_2_bias = lattice.weight @right_2_bias {storage_key = "right.2.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %right_2_features = lattice.layer_norm %right_2_features_in, %right_2_weight, %right_2_bias {eps = 0.00001 : f32} : (tensor<?x4xf32>, !lattice.weight<channel, f32>, !lattice.weight<bias, f32>) -> tensor<?x4xf32>
    %right_2 = lattice.sparse.with_features %right_1, %right_2_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x4xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %cat_1 = lattice.sparse.cat %left_2, %right_2 {join = #lattice.join<inner>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %tail_0_weight = lattice.weight @tail_0_weight {storage_key = "tail.0.weight", layout = #lattice.weight_layout<conv3d_o_zyx_i>, packing = #lattice.packing<dense>} : !lattice.weight<conv3d, f32>
    %tail_0 = lattice.conv3d %cat_1, %tail_0_weight {kernel_size = array<i64: 1, 1, 1>, stride = array<i64: 1, 1, 1>, padding = array<i64: 0, 0, 0>, dilation = array<i64: 1, 1, 1>} : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, !lattice.weight<conv3d, f32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose4, %tail_1_features_in, %tail_1_active = lattice.sparse.decompose %tail_0 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x5xf32>, tensor<1xi32>)
    %tail_1_features = lattice.activation %tail_1_features_in {kind = #lattice.activation<leaky_relu>, approximate = #lattice.gelu_approx<none>, alpha = 0.01 : f32, beta = 1.0 : f32, threshold = 20.0 : f32} : (tensor<?x5xf32>) -> tensor<?x5xf32>
    %tail_1 = lattice.sparse.with_features %tail_0, %tail_1_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x5xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    %sparse_decompose5, %tail_2_features_in, %tail_2_active = lattice.sparse.decompose %tail_1 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32> -> (tensor<?x4xi32>, tensor<?x5xf32>, tensor<1xi32>)
    %tail_2_weight = lattice.weight @tail_2_weight {storage_key = "tail.2.weight", layout = #lattice.weight_layout<channel_c>, packing = #lattice.packing<dense>} : !lattice.weight<channel, f32>
    %tail_2_bias = lattice.weight @tail_2_bias {storage_key = "tail.2.bias", layout = #lattice.weight_layout<bias_c>, packing = #lattice.packing<dense>} : !lattice.weight<bias, f32>
    %tail_2_features = lattice.layer_norm %tail_2_features_in, %tail_2_weight, %tail_2_bias {eps = 0.00001 : f32} : (tensor<?x5xf32>, !lattice.weight<channel, f32>, !lattice.weight<bias, f32>) -> tensor<?x5xf32>
    %tail_2 = lattice.sparse.with_features %tail_1, %tail_2_features  : (!lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>, tensor<?x5xf32>) -> !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
    return %tail_2 : !lattice.sparse_tensor<rank = 3, coord = batch_x_y_z, feature = row_channel, dtype = f32>
  }
}

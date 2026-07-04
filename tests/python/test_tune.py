from collections import defaultdict

import pytest
import torch_lattice
import torch
from torch_lattice.nn import functional as F
from torch_lattice.nn.functional.conv.func.fetch_on_demand import (
    FetchOnDemandConvolutionFuntion,
)
from torch_lattice.nn.functional.conv.func.implicit_gemm import (
    ImplicitGEMMConvolutionFuntion,
    _resolve_wgrad_split_k,
)
from torch_lattice.utils import tune as tune_module


def test_recursive_apply_preserves_tuple_inputs(monkeypatch):
    class DummySparseTensor:
        def __init__(self, value):
            self.value = value

    monkeypatch.setattr(tune_module, "SparseTensor", DummySparseTensor)

    inputs = {
        "points": (
            DummySparseTensor(1),
            [DummySparseTensor(2), "unchanged"],
        )
    }
    output = tune_module.recursive_apply(
        inputs, lambda tensor: DummySparseTensor(tensor.value + 10)
    )

    assert isinstance(output["points"], tuple)
    assert output["points"][0].value == 11
    assert output["points"][1][0].value == 12
    assert output["points"][1][1] == "unchanged"


def test_dataflow_selector_profiles_fetch_on_demand_fused_and_no_fusion(monkeypatch):
    seen = []

    def fake_set_group_config(model, names, config):
        seen.append((config.dataflow, config.ifsort, config.split_mask_num, config.FOD_fusion))

    monkeypatch.setattr(tune_module, "set_group_config", fake_set_group_config)
    monkeypatch.setattr(tune_module, "clear_model_config", lambda model: None)
    monkeypatch.setattr(tune_module, "clear_tensor_cache", lambda inputs: inputs)
    monkeypatch.setattr(tune_module, "torch_lattice_tune_timer", lambda model, inputs, tune_with_bwd: (1.0, 0.0))

    group_to_name = defaultdict(list)
    group_to_name[((1, 1, 1), (3, 3, 3), (1, 1, 1), (1, 1, 1))].append("conv")
    dataflow_all = defaultdict(lambda: defaultdict(tune_module.StableTimeAccumulator))

    tune_module.dataflow_selector(
        model=object(),
        inputs=object(),
        dataflow_range=[F.Dataflow.FetchOnDemand],
        group_to_name=group_to_name,
        dataflow_all=dataflow_all,
        tune_with_bwd=False,
    )

    assert seen == [
        (F.Dataflow.FetchOnDemand, False, 1, True),
        (F.Dataflow.FetchOnDemand, False, 1, False),
    ]
    assert set(dataflow_all[next(iter(group_to_name))].keys()) == {
        (F.Dataflow.FetchOnDemand, True),
        (F.Dataflow.FetchOnDemand, False),
    }


def test_profile_model_pruned_fetch_on_demand_uses_selected_fusion(monkeypatch):
    seen = []

    def fake_set_group_config(model, names, config):
        seen.append(config.FOD_fusion)

    monkeypatch.setattr(tune_module, "set_group_config", fake_set_group_config)
    monkeypatch.setattr(tune_module, "clear_model_config", lambda model: None)
    monkeypatch.setattr(tune_module, "clear_tensor_cache", lambda inputs: inputs)
    monkeypatch.setattr(tune_module, "torch_lattice_tune_timer", lambda model, inputs, tune_with_bwd: (1.0, 0.0))

    group_idx = ((1, 1, 1), (3, 3, 3), (1, 1, 1), (1, 1, 1))
    group_to_name = defaultdict(list)
    group_to_name[group_idx].append("conv")
    configs_all = defaultdict(lambda: defaultdict(tune_module.StableTimeAccumulator))

    tune_module.profile_model(
        model=object(),
        inputs=object(),
        dataflow_range=[F.Dataflow.ImplicitGEMM, F.Dataflow.FetchOnDemand],
        dataflow_prune=True,
        group_to_name=group_to_name,
        configs_all=configs_all,
        group_dataflow={
            group_idx: {
                "dataflow": F.Dataflow.FetchOnDemand,
                "FOD_fusion": False,
            }
        },
        tune_with_bwd=False,
    )

    assert seen == [False]
    assert len(configs_all[group_idx]) == 1
    only_config = next(iter(configs_all[group_idx]))
    assert only_config[4:] == (F.Dataflow.FetchOnDemand, False, False, False, "auto")


def test_dataflow_prune_selection_can_choose_fetch_on_demand_no_fusion():
    group_idx = ((1, 1, 1), (3, 3, 3), (1, 1, 1), (1, 1, 1))
    dataflow_all = defaultdict(lambda: defaultdict(tune_module.StableTimeAccumulator))
    dataflow_all[group_idx][(F.Dataflow.ImplicitGEMM, None)].stable_add(0.50, 0.0)
    dataflow_all[group_idx][(F.Dataflow.FetchOnDemand, True)].stable_add(5.00, 0.0)
    dataflow_all[group_idx][(F.Dataflow.FetchOnDemand, False)].stable_add(0.10, 0.0)

    time_min = -1.0
    for dataflow, FOD_fusion in dataflow_all[group_idx]:
        total_time = dataflow_all[group_idx][(dataflow, FOD_fusion)].get_total_time()
        if time_min < 0 or time_min > total_time:
            time_min = total_time
            dataflow_best = dataflow
            FOD_fusion_best = FOD_fusion

    assert dataflow_best == F.Dataflow.FetchOnDemand
    assert FOD_fusion_best is False


def test_default_conv_config_does_not_share_mutable_state():
    config = F.conv_config.get_default_conv_config()
    config.dataflow = F.Dataflow.FetchOnDemand
    config.ifsort = True
    config.split_mask_num = 99

    fresh = F.conv_config.get_default_conv_config()

    assert fresh.dataflow == F.Dataflow.ImplicitGEMM
    assert fresh.ifsort is False
    assert fresh.split_mask_num == 1
    assert fresh.wgrad_split_k == "auto"
    assert fresh.FOD_fusion is False


def test_tune_default_dataflows_include_fod_for_forward_only(monkeypatch, tmp_path):
    seen = []

    class DummyParameter:
        device = "cuda:0"

    class DummyModel:
        def parameters(self):
            return iter([DummyParameter()])

        def named_modules(self):
            return iter(())

        def cuda(self):
            return self

        def __call__(self, inputs):
            return inputs

        def __call__(self, inputs):
            return inputs

    def fake_profile_model(
        model,
        inputs,
        dataflow_range,
        dataflow_prune,
        group_to_name,
        configs_all,
        group_dataflow,
        tune_with_bwd,
    ):
        seen.append((tuple(dataflow_range), tune_with_bwd))

    monkeypatch.setattr(tune_module, "profile_model", fake_profile_model)
    monkeypatch.setattr(tune_module, "recursive_apply", lambda value, func: value)
    monkeypatch.setattr(torch_lattice.backends, "benchmark", True)

    tune_module.tune(
        DummyModel(),
        data_loader=[object()],
        n_samples=1,
        save_dir=str(tmp_path),
        tune_tag="forward",
        force_retune=True,
        verbose=False,
        skip_warning=True,
    )
    tune_module.tune(
        DummyModel(),
        data_loader=[object()],
        n_samples=1,
        save_dir=str(tmp_path),
        tune_tag="backward",
        force_retune=True,
        tune_with_bwd=True,
        verbose=False,
        skip_warning=True,
    )

    assert seen[0] == (
        (F.Dataflow.ImplicitGEMM, F.Dataflow.FetchOnDemand),
        False,
    )
    assert seen[1] == (
        (F.Dataflow.ImplicitGEMM,),
        True,
    )


def test_tune_backward_can_explicitly_include_fetch_on_demand(monkeypatch, tmp_path):
    seen = []

    class DummyParameter:
        device = "cuda:0"

    class DummyModel:
        def parameters(self):
            return iter([DummyParameter()])

        def named_modules(self):
            return iter(())

        def cuda(self):
            return self

        def __call__(self, inputs):
            return inputs

    def fake_profile_model(
        model,
        inputs,
        dataflow_range,
        dataflow_prune,
        group_to_name,
        configs_all,
        group_dataflow,
        tune_with_bwd,
    ):
        seen.append((tuple(dataflow_range), tune_with_bwd))

    monkeypatch.setattr(tune_module, "profile_model", fake_profile_model)
    monkeypatch.setattr(tune_module, "recursive_apply", lambda value, func: value)
    monkeypatch.setattr(torch_lattice.backends, "benchmark", True)

    tune_module.tune(
        DummyModel(),
        data_loader=[object()],
        n_samples=1,
        save_dir=str(tmp_path),
        tune_tag="explicit_backward",
        force_retune=True,
        tune_with_bwd=True,
        dataflow_range=[F.Dataflow.ImplicitGEMM, F.Dataflow.FetchOnDemand],
        verbose=False,
        skip_warning=True,
    )

    assert seen == [
        ((F.Dataflow.ImplicitGEMM, F.Dataflow.FetchOnDemand), True),
    ]


def test_profile_model_training_profiles_fetch_on_demand_variants(monkeypatch):
    seen = []

    def fake_set_group_config(model, names, config):
        seen.append((config.dataflow, config.FOD_fusion))

    monkeypatch.setattr(tune_module, "set_group_config", fake_set_group_config)
    monkeypatch.setattr(tune_module, "clear_model_config", lambda model: None)
    monkeypatch.setattr(tune_module, "clear_tensor_cache", lambda inputs: inputs)
    monkeypatch.setattr(tune_module, "torch_lattice_tune_timer", lambda model, inputs, tune_with_bwd: (1.0, 0.5))

    group_idx = ((1, 1, 1), (3, 3, 3), (1, 1, 1), (1, 1, 1))
    group_to_name = defaultdict(list)
    group_to_name[group_idx].append("conv")
    configs_all = defaultdict(lambda: defaultdict(tune_module.StableTimeAccumulator))

    tune_module.profile_model(
        model=object(),
        inputs=object(),
        dataflow_range=[F.Dataflow.FetchOnDemand],
        dataflow_prune=False,
        group_to_name=group_to_name,
        configs_all=configs_all,
        group_dataflow={},
        tune_with_bwd=True,
    )

    assert seen == [
        (F.Dataflow.FetchOnDemand, True),
        (F.Dataflow.FetchOnDemand, False),
    ]
    assert set(config[4:] for config in configs_all[group_idx]) == {
        (F.Dataflow.FetchOnDemand, False, True, False, "auto"),
        (F.Dataflow.FetchOnDemand, False, False, False, "auto"),
    }


def test_profile_model_training_profiles_implicit_wgrad_split_k(monkeypatch):
    seen = []

    def fake_set_group_config(model, names, config):
        seen.append((config.ifsort, config.split_mask_num, config.split_mask_num_bwd, config.wgrad_split_k))

    monkeypatch.setattr(tune_module, "set_group_config", fake_set_group_config)
    monkeypatch.setattr(tune_module, "clear_model_config", lambda model: None)
    monkeypatch.setattr(tune_module, "clear_tensor_cache", lambda inputs: inputs)
    monkeypatch.setattr(tune_module, "torch_lattice_tune_timer", lambda model, inputs, tune_with_bwd: (1.0, 0.5))

    group_idx = ((1, 1, 1), (3, 3, 3), (1, 1, 1), (1, 1, 1))
    group_to_name = defaultdict(list)
    group_to_name[group_idx].append("conv")
    configs_all = defaultdict(lambda: defaultdict(tune_module.StableTimeAccumulator))

    tune_module.profile_model(
        model=object(),
        inputs=object(),
        dataflow_range=[F.Dataflow.ImplicitGEMM],
        dataflow_prune=False,
        group_to_name=group_to_name,
        configs_all=configs_all,
        group_dataflow={},
        tune_with_bwd=True,
    )

    sorted_profiles = [item for item in seen if item[0]]
    unsorted_profiles = [item for item in seen if not item[0]]
    assert set(item[3] for item in sorted_profiles) == {8, 16, 32, 64}
    assert set(item[3] for item in unsorted_profiles) == {8, 16, 32, 64}
    assert any(config[-1] == 8 for config in configs_all[group_idx])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_implicit_gemm_backward_passes_configured_wgrad_split_k(monkeypatch):
    seen = []

    def fake_dgrad(
        grad_output,
        weight,
        reorder_out_in_map_bwd,
        reduced_sorted_mask_bwd_dgrad,
        reorder_loc_bwd,
        input_size,
        input_channels,
        allow_tf32,
        allow_fp16,
    ):
        return torch.zeros(input_size, input_channels, device=grad_output.device, dtype=grad_output.dtype)

    def fake_wgrad(
        grad_output,
        input,
        reorder_out_in_map_bwd,
        reduced_sorted_mask_bwd_wgrad,
        reorder_loc_bwd,
        split_k_iters,
        allow_tf32,
        allow_fp16,
    ):
        seen.append(split_k_iters)
        kernel_volume = 3
        return torch.zeros(
            kernel_volume * input.size(1),
            grad_output.size(1),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )

    monkeypatch.setattr(
        torch_lattice.backend, "conv_forward_implicit_gemm_sorted_cuda", fake_dgrad
    )
    monkeypatch.setattr(
        torch_lattice.backend, "conv_backward_wgrad_implicit_gemm_sorted_cuda", fake_wgrad
    )

    class Ctx:
        needs_input_grad = (True, True, False, False, False)

    ctx = Ctx()
    input = torch.randn(5, 4, device="cuda", dtype=torch.float16)
    weight = torch.randn(3, 4, 6, device="cuda", dtype=torch.float16)
    placeholder = torch.empty(0, device="cuda", dtype=torch.int32)
    ctx.for_backwards = (
        input,
        weight,
        None,
        placeholder,
        placeholder,
        placeholder,
        placeholder,
        False,
        8,
        False,
        None,
        weight.size(0),
        True,
    )

    grad_output = torch.randn(5, 6, device="cuda", dtype=torch.float16)
    ImplicitGEMMConvolutionFuntion.backward(ctx, grad_output)

    assert seen == [8]


def test_implicit_gemm_auto_wgrad_split_k_uses_kernel_volume_heuristic():
    config = F.conv_config.get_default_conv_config().copy()

    assert _resolve_wgrad_split_k(config, 3) == 64
    assert _resolve_wgrad_split_k(config, 9) == 16
    assert _resolve_wgrad_split_k(config, 27) == 8
    assert _resolve_wgrad_split_k(config, 3, ifsort=True) == 32
    assert _resolve_wgrad_split_k(config, 27, ifsort=True) == 32

    config.wgrad_split_k = 8
    assert _resolve_wgrad_split_k(config, 27) == 8
    assert _resolve_wgrad_split_k(config, 27, ifsort=True) == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_implicit_gemm_backward_respects_unsorted_forward_dispatch(monkeypatch):
    seen = []

    def fake_dgrad_unsorted(
        grad_output,
        weight,
        out_in_map_bwd,
        input_size,
        input_channels,
        allow_tf32,
        allow_fp16,
    ):
        seen.append("dgrad_unsorted")
        return torch.zeros(
            input_size, input_channels, device=grad_output.device, dtype=grad_output.dtype
        )

    def fake_wgrad_unsorted(
        grad_output,
        input,
        out_in_map_bwd,
        split_k_iters,
        allow_tf32,
        allow_fp16,
    ):
        seen.append(("wgrad_unsorted", split_k_iters))
        return torch.zeros(
            3 * input.size(1),
            grad_output.size(1),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )

    def fail_sorted(*args, **kwargs):
        raise AssertionError("unsorted forward config should use unsorted backward kernels")

    monkeypatch.setattr(
        torch_lattice.backend, "conv_forward_implicit_gemm_cuda", fake_dgrad_unsorted
    )
    monkeypatch.setattr(
        torch_lattice.backend, "conv_backward_wgrad_implicit_gemm_cuda", fake_wgrad_unsorted
    )
    monkeypatch.setattr(
        torch_lattice.backend, "conv_forward_implicit_gemm_sorted_cuda", fail_sorted
    )
    monkeypatch.setattr(
        torch_lattice.backend, "conv_backward_wgrad_implicit_gemm_sorted_cuda", fail_sorted
    )

    class Ctx:
        needs_input_grad = (True, True, False, False, False)

    ctx = Ctx()
    input = torch.randn(5, 4, device="cuda", dtype=torch.float16)
    weight = torch.randn(3, 4, 6, device="cuda", dtype=torch.float16)
    out_in_map = torch.empty(0, device="cuda", dtype=torch.int32)
    ctx.for_backwards = (
        input,
        weight,
        out_in_map,
        None,
        None,
        None,
        None,
        False,
        64,
        False,
        None,
        weight.size(0),
        False,
    )

    grad_output = torch.randn(5, 6, device="cuda", dtype=torch.float16)
    ImplicitGEMMConvolutionFuntion.backward(ctx, grad_output)

    assert seen == ["dgrad_unsorted", ("wgrad_unsorted", 64)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_implicit_gemm_center_only_uses_matmul_fast_path(monkeypatch):
    def fail_backend(*args, **kwargs):
        raise AssertionError("center-only ImplicitGEMM should not launch backend kernels")

    monkeypatch.setattr(
        torch_lattice.backend, "conv_forward_implicit_gemm_cuda", fail_backend
    )
    monkeypatch.setattr(
        torch_lattice.backend, "conv_forward_implicit_gemm_sorted_cuda", fail_backend
    )

    points = 17
    in_channels = 16
    out_channels = 8
    kernel_volume = 27
    mid_kernel = kernel_volume // 2
    input = torch.randn(points, in_channels, device="cuda", dtype=torch.float16)
    weight = torch.randn(
        kernel_volume, in_channels, out_channels, device="cuda", dtype=torch.float16
    )
    kmap = {
        "sizes": (points, points),
        "out_in_map": None,
        "reorder_out_in_map": None,
        "reduced_sorted_mask": None,
        "reorder_loc": None,
        "out_in_map_bwd": None,
        "reorder_out_in_map_bwd": None,
        "reduced_sorted_mask_bwd_wgrad": None,
        "reduced_sorted_mask_bwd_dgrad": None,
        "reorder_loc_bwd": None,
        "IGEMM_center_only": True,
    }
    config = F.conv_config.get_default_conv_config().copy()
    config.ifsort = False

    output = ImplicitGEMMConvolutionFuntion.apply(input, weight, kmap, config, False)

    torch.testing.assert_close(output, input.matmul(weight[mid_kernel]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_implicit_gemm_center_only_backward_uses_matmul_fast_path(monkeypatch):
    def fail_backend(*args, **kwargs):
        raise AssertionError("center-only ImplicitGEMM backward should not launch backend kernels")

    monkeypatch.setattr(
        torch_lattice.backend, "conv_forward_implicit_gemm_cuda", fail_backend
    )
    monkeypatch.setattr(
        torch_lattice.backend, "conv_forward_implicit_gemm_sorted_cuda", fail_backend
    )
    monkeypatch.setattr(
        torch_lattice.backend, "conv_backward_wgrad_implicit_gemm_cuda", fail_backend
    )
    monkeypatch.setattr(
        torch_lattice.backend, "conv_backward_wgrad_implicit_gemm_sorted_cuda", fail_backend
    )

    points = 23
    in_channels = 8
    out_channels = 12
    kernel_volume = 27
    mid_kernel = kernel_volume // 2
    input = torch.randn(
        points, in_channels, device="cuda", dtype=torch.float32, requires_grad=True
    )
    weight = torch.randn(
        kernel_volume,
        in_channels,
        out_channels,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    ref_input = input.detach().clone().requires_grad_(True)
    ref_weight_mid = weight.detach()[mid_kernel].clone().requires_grad_(True)
    kmap = {
        "sizes": (points, points),
        "out_in_map": None,
        "reorder_out_in_map": None,
        "reduced_sorted_mask": None,
        "reorder_loc": None,
        "out_in_map_bwd": None,
        "reorder_out_in_map_bwd": None,
        "reduced_sorted_mask_bwd_wgrad": None,
        "reduced_sorted_mask_bwd_dgrad": None,
        "reorder_loc_bwd": None,
        "IGEMM_center_only": True,
    }
    config = F.conv_config.get_default_conv_config().copy()
    config.ifsort = False

    output = ImplicitGEMMConvolutionFuntion.apply(input, weight, kmap, config, False)
    ref_output = ref_input.matmul(ref_weight_mid)
    grad_output = torch.randn_like(output)

    output.backward(grad_output)
    ref_output.backward(grad_output)

    expected_weight_grad = torch.zeros_like(weight)
    expected_weight_grad[mid_kernel] = ref_weight_mid.grad

    torch.testing.assert_close(input.grad, ref_input.grad)
    torch.testing.assert_close(weight.grad, expected_weight_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_implicit_gemm_active_weight_cache_invalidates_on_weight_update(monkeypatch):
    seen_weights = []

    def fake_backend(input, weight, out_in_map, num_out_feats, num_out_channels, *args):
        seen_weights.append(weight)
        return input[:, :num_out_channels].clone()

    monkeypatch.setattr(
        torch_lattice.backend, "conv_forward_implicit_gemm_cuda", fake_backend
    )

    points = 8
    in_channels = 4
    out_channels = 4
    input = torch.randn(points, in_channels, device="cuda", dtype=torch.float16)
    weight = torch.randn(
        27, in_channels, out_channels, device="cuda", dtype=torch.float16
    )
    active_kernel_offsets = torch.tensor([12, 13, 14], device="cuda")
    kmap = {
        "sizes": (points, points),
        "out_in_map": torch.full((points, 3), -1, device="cuda", dtype=torch.int32),
        "reorder_out_in_map": None,
        "reduced_sorted_mask": None,
        "reorder_loc": None,
        "out_in_map_bwd": None,
        "reorder_out_in_map_bwd": None,
        "reduced_sorted_mask_bwd_wgrad": None,
        "reduced_sorted_mask_bwd_dgrad": None,
        "reorder_loc_bwd": None,
        "active_kernel_offsets": active_kernel_offsets,
    }
    config = F.conv_config.get_default_conv_config().copy()
    config.ifsort = False

    ImplicitGEMMConvolutionFuntion.apply(input, weight, kmap, config, False)
    ImplicitGEMMConvolutionFuntion.apply(input, weight, kmap, config, False)
    with torch.no_grad():
        weight.add_(0.125)
    ImplicitGEMMConvolutionFuntion.apply(input, weight, kmap, config, False)

    assert seen_weights[0] is seen_weights[1]
    assert seen_weights[2] is not seen_weights[1]
    assert kmap["_active_weight_cache"][0][1] == weight._version
    torch.testing.assert_close(
        seen_weights[2], weight.index_select(0, active_kernel_offsets).contiguous()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fetch_on_demand_center_only_uses_matmul_fast_path(monkeypatch):
    def fail_backend(*args, **kwargs):
        raise AssertionError("center-only FOD should not launch backend kernels")

    monkeypatch.setattr(
        torch_lattice.backend, "conv_forward_fetch_on_demand_cuda", fail_backend
    )
    monkeypatch.setattr(
        torch_lattice.backend,
        "conv_forward_fetch_on_demand_no_fusion_cuda",
        fail_backend,
    )

    points = 17
    in_channels = 16
    out_channels = 8
    kernel_volume = 27
    mid_kernel = kernel_volume // 2
    input = torch.randn(points, in_channels, device="cuda", dtype=torch.float16)
    weight = torch.randn(
        kernel_volume, in_channels, out_channels, device="cuda", dtype=torch.float16
    )
    nbsizes_cpu = torch.zeros(kernel_volume, dtype=torch.int32)
    nbsizes_cpu[mid_kernel] = points
    ids = torch.arange(points, device="cuda", dtype=torch.int32)
    kmap = {
        "nbmaps": torch.stack([ids, ids]),
        "nbsizes": nbsizes_cpu.to("cuda"),
        "nbsizes_cpu": nbsizes_cpu,
        "FOD_center_only": True,
        "sizes": (points, points),
    }
    config = F.conv_config.get_default_conv_config().copy()
    config.FOD_fusion = False

    output = FetchOnDemandConvolutionFuntion.apply(input, weight, kmap, config, False)

    torch.testing.assert_close(output, input.matmul(weight[mid_kernel]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fetch_on_demand_center_only_backward_uses_matmul_fast_path(monkeypatch):
    def fail_backward(*args, **kwargs):
        raise AssertionError("center-only FOD backward should not use gather-scatter")

    monkeypatch.setattr(
        torch_lattice.backend, "conv_backward_gather_scatter_cuda", fail_backward
    )

    points = 23
    in_channels = 8
    out_channels = 12
    kernel_volume = 27
    mid_kernel = kernel_volume // 2
    input = torch.randn(
        points, in_channels, device="cuda", dtype=torch.float32, requires_grad=True
    )
    weight = torch.randn(
        kernel_volume,
        in_channels,
        out_channels,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    ref_input = input.detach().clone().requires_grad_(True)
    ref_weight_mid = weight.detach()[mid_kernel].clone().requires_grad_(True)
    nbsizes_cpu = torch.zeros(kernel_volume, dtype=torch.int32)
    nbsizes_cpu[mid_kernel] = points
    ids = torch.arange(points, device="cuda", dtype=torch.int32)
    kmap = {
        "nbmaps": torch.stack([ids, ids]),
        "nbsizes": nbsizes_cpu.to("cuda"),
        "nbsizes_cpu": nbsizes_cpu,
        "FOD_center_only": True,
        "sizes": (points, points),
    }
    config = F.conv_config.get_default_conv_config().copy()
    config.FOD_fusion = False

    output = FetchOnDemandConvolutionFuntion.apply(input, weight, kmap, config, False)
    ref_output = ref_input.matmul(ref_weight_mid)
    grad_output = torch.randn_like(output)

    output.backward(grad_output)
    ref_output.backward(grad_output)

    expected_weight_grad = torch.zeros_like(weight)
    expected_weight_grad[mid_kernel] = ref_weight_mid.grad

    torch.testing.assert_close(input.grad, ref_input.grad)
    torch.testing.assert_close(weight.grad, expected_weight_grad)

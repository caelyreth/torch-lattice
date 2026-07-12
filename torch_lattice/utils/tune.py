import functools
import os
import time
from collections import defaultdict
from typing import Callable, DefaultDict, Iterable, Iterator, List, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import torch_lattice
from torch_lattice import SparseTensor
from torch_lattice.nn import (
    Conv3d,
    ConvTranspose3d,
    GenerativeConvTranspose3d,
    SubmConv3d,
)
from torch_lattice.utils import make_ntuple
from torch_lattice.nn import functional as F

__all__ = ["tune"]

_TUNABLE_CONV_TYPES = (
    Conv3d,
    SubmConv3d,
    ConvTranspose3d,
    GenerativeConvTranspose3d,
)


_WGRAD_SPLIT_K_RANGE = (8, 16, 32, 64)


class StableTimeAccumulator:
    def __init__(self):
        self.fwd_trial = 0
        self.ave_fwd_time = 0.0
        self.bwd_trial = 0
        self.ave_bwd_time = 0.0

    def stable_add(self, cur_fwd_time: float, cur_bwd_time: float):
        if cur_fwd_time > 0:
            if self.fwd_trial == 0:
                self.ave_fwd_time = cur_fwd_time
                self.fwd_trial += 1
            else:
                if cur_fwd_time <= 5 * self.ave_fwd_time:
                    self.ave_fwd_time = (
                        (self.fwd_trial * self.ave_fwd_time) + cur_fwd_time
                    ) / (self.fwd_trial + 1)
                    self.fwd_trial += 1
        if cur_bwd_time > 0:
            if self.bwd_trial == 0:
                self.ave_bwd_time = cur_bwd_time
                self.bwd_trial += 1
            else:
                if cur_bwd_time <= 5 * self.ave_bwd_time:
                    self.ave_bwd_time = (
                        (self.bwd_trial * self.ave_bwd_time) + cur_bwd_time
                    ) / (self.bwd_trial + 1)
                    self.bwd_trial += 1

    def get_total_time(self):
        return self.ave_fwd_time + self.ave_bwd_time


def recursive_apply(x, func):
    if isinstance(x, dict):
        return {k: recursive_apply(v, func) for k, v in x.items()}
    if isinstance(x, list):
        return [recursive_apply(v, func) for v in x]
    if isinstance(x, tuple):
        return tuple(recursive_apply(v, func) for v in x)
    if isinstance(x, SparseTensor):
        temp = func(x)
        return temp if isinstance(temp, SparseTensor) else x
    return x


def _dataflow_prune_key(config):
    if config.dataflow == F.Dataflow.FetchOnDemand:
        return (config.dataflow, config.FOD_fusion)
    return (config.dataflow, None)


def _config_key(config):
    return (
        config.epsilon,
        config.mm_thresh,
        config.split_mask_num,
        config.split_mask_num_bwd,
        config.dataflow,
        config.ifsort,
        config.FOD_fusion,
        config.get("IGEMM_center_only", False),
        config.wgrad_split_k,
    )


def clear_tensor_cache(inputs: SparseTensor):
    return recursive_apply(inputs, lambda x: x.coord_manager.clear_relations())


def clear_model_config(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, _TUNABLE_CONV_TYPES):
            module._config = F.conv_config.get_default_conv_config()


def set_group_config(model: nn.Module, names: list, config: Dict):
    for name, module in model.named_modules():
        if isinstance(module, _TUNABLE_CONV_TYPES):
            if name in names:
                module._config = config.copy()


def torch_lattice_tune_timer(
    model: nn.Module,
    inputs: SparseTensor,
    tune_with_bwd: bool,
) -> float:
    fwd_time = 0.0
    bwd_time = 0.0
    torch.cuda.synchronize()
    st = time.time()
    outputs = model(inputs)
    torch.cuda.synchronize()
    ed = time.time()
    fwd_time = ed - st

    if tune_with_bwd:
        top_grad = torch.randn_like(outputs) * 1e-3
        torch.cuda.synchronize()
        st_bp = time.time()
        outputs.backward(top_grad)
        torch.cuda.synchronize()
        ed_bp = time.time()
        bwd_time = ed_bp - st_bp

    return fwd_time, bwd_time


def dataflow_selector(
    model: nn.Module,
    inputs: SparseTensor,
    dataflow_range: List,
    group_to_name: DefaultDict[Tuple[Any, ...], List],
    dataflow_all: DefaultDict[
        Tuple[Any, ...], DefaultDict[Tuple[Any, ...], StableTimeAccumulator]
    ],
    tune_with_bwd: bool,
) -> None:

    for group_idx, names in group_to_name.items():
        # Set all configs to default
        clear_model_config(model)
        dummy_config = F.conv_config.get_default_conv_config().copy()

        # Dataflow 1: ImplicitGEMM (Test 2 representative examples)
        if F.Dataflow.ImplicitGEMM in dataflow_range:
            # Setting 1: ImplicitGEMM-unsort
            dummy_config.dataflow, dummy_config.ifsort, dummy_config.split_mask_num = (
                F.Dataflow.ImplicitGEMM,
                False,
                1,
            )
            set_group_config(model, names, dummy_config)
            inputs = clear_tensor_cache(inputs)
            fwd_duration, bwd_duration = torch_lattice_tune_timer(
                model, inputs, tune_with_bwd
            )
            dataflow_all[group_idx][_dataflow_prune_key(dummy_config)].stable_add(
                fwd_duration, bwd_duration
            )

            # Setting 2: ImplicitGEMM-sort(split=3)
            dummy_config.dataflow, dummy_config.ifsort, dummy_config.split_mask_num = (
                F.Dataflow.ImplicitGEMM,
                True,
                3,
            )
            set_group_config(model, names, dummy_config)
            inputs = clear_tensor_cache(inputs)
            fwd_duration, bwd_duration = torch_lattice_tune_timer(
                model, inputs, tune_with_bwd
            )
            dataflow_all[group_idx][_dataflow_prune_key(dummy_config)].stable_add(
                fwd_duration, bwd_duration
            )

        # Dataflow 2: Fetch-On-Demand
        if F.Dataflow.FetchOnDemand in dataflow_range:
            dummy_config.dataflow, dummy_config.ifsort, dummy_config.split_mask_num = (
                F.Dataflow.FetchOnDemand,
                False,
                1,
            )
            for FOD_fusion in [True, False]:
                dummy_config.FOD_fusion = FOD_fusion
                set_group_config(model, names, dummy_config)
                inputs = clear_tensor_cache(inputs)
                fwd_duration, bwd_duration = torch_lattice_tune_timer(
                    model, inputs, tune_with_bwd
                )
                dataflow_all[group_idx][_dataflow_prune_key(dummy_config)].stable_add(
                    fwd_duration, bwd_duration
                )

        # Dataflow 3: Gather-Scatter (Deprecated by default)
        if F.Dataflow.GatherScatter in dataflow_range:
            dummy_config.dataflow, dummy_config.ifsort, dummy_config.split_mask_num = (
                F.Dataflow.GatherScatter,
                False,
                1,
            )
            set_group_config(model, names, dummy_config)
            inputs = clear_tensor_cache(inputs)
            fwd_duration, bwd_duration = torch_lattice_tune_timer(
                model, inputs, tune_with_bwd
            )
            dataflow_all[group_idx][_dataflow_prune_key(dummy_config)].stable_add(
                fwd_duration, bwd_duration
            )


# @torch.no_grad()
def profile_model(
    model: nn.Module,
    inputs: SparseTensor,
    dataflow_range: List,
    dataflow_prune: bool,
    group_to_name: DefaultDict[Tuple[Any, ...], List],
    configs_all: DefaultDict[
        Tuple[Any, ...], DefaultDict[Tuple[Any, ...], StableTimeAccumulator]
    ],
    group_dataflow: Dict,
    tune_with_bwd: bool,
) -> None:

    for group_idx, names in group_to_name.items():
        # Set all configs to default
        clear_model_config(model)
        dummy_config = F.conv_config.get_default_conv_config().copy()
        if dataflow_prune:
            pruned_dataflow = group_dataflow[group_idx]
            local_dataflow_range = [pruned_dataflow["dataflow"]]
            pruned_FOD_fusion = pruned_dataflow.get("FOD_fusion")
        else:
            local_dataflow_range = dataflow_range
            pruned_FOD_fusion = None

        if F.Dataflow.ImplicitGEMM in local_dataflow_range:
            # Implicit-GEMM. Tune whether to sort & split_mask_num.
            dummy_config.dataflow = F.Dataflow.ImplicitGEMM

            # Stage 1: test unsort fwd
            dummy_config.ifsort = False
            dummy_config.IGEMM_center_only = False
            if not tune_with_bwd:
                for IGEMM_center_only in (False, True):
                    dummy_config.IGEMM_center_only = IGEMM_center_only
                    dummy_config.split_mask_num = 1
                    set_group_config(model, names, dummy_config)
                    inputs = clear_tensor_cache(inputs)
                    fwd_duration, bwd_duration = torch_lattice_tune_timer(
                        model, inputs, tune_with_bwd
                    )
                    configs_all[group_idx][_config_key(dummy_config)].stable_add(
                        fwd_duration, bwd_duration
                    )
            else:
                dummy_config.IGEMM_center_only = False
                for split_mask_num_bwd in range(1, 5):
                    dummy_config.split_mask_num_bwd = split_mask_num_bwd
                    for wgrad_split_k in _WGRAD_SPLIT_K_RANGE:
                        dummy_config.wgrad_split_k = wgrad_split_k
                        set_group_config(model, names, dummy_config)
                        inputs = clear_tensor_cache(inputs)
                        fwd_duration, bwd_duration = torch_lattice_tune_timer(
                            model, inputs, tune_with_bwd
                        )
                        configs_all[group_idx][_config_key(dummy_config)].stable_add(
                            fwd_duration, bwd_duration
                        )

            # Stage 2: test sort fwd
            dummy_config.ifsort = True
            dummy_config.IGEMM_center_only = False
            if not tune_with_bwd:
                for IGEMM_center_only in (False, True):
                    dummy_config.IGEMM_center_only = IGEMM_center_only
                    for split_mask_num in range(1, 5):
                        dummy_config.split_mask_num = split_mask_num
                        set_group_config(model, names, dummy_config)
                        inputs = clear_tensor_cache(inputs)
                        fwd_duration, bwd_duration = torch_lattice_tune_timer(
                            model, inputs, tune_with_bwd
                        )
                        configs_all[group_idx][_config_key(dummy_config)].stable_add(
                            fwd_duration, bwd_duration
                        )
            else:
                dummy_config.IGEMM_center_only = False
                for split_mask_num in range(1, 5):
                    dummy_config.split_mask_num = split_mask_num
                    dummy_config.split_mask_num_bwd = split_mask_num
                    for wgrad_split_k in _WGRAD_SPLIT_K_RANGE:
                        dummy_config.wgrad_split_k = wgrad_split_k
                        set_group_config(model, names, dummy_config)
                        inputs = clear_tensor_cache(inputs)
                        fwd_duration, bwd_duration = torch_lattice_tune_timer(
                            model, inputs, tune_with_bwd
                        )
                        for iter in range(1, 5):
                            fwd_config = dummy_config.copy()
                            fwd_config.split_mask_num_bwd = iter
                            configs_all[group_idx][_config_key(fwd_config)].stable_add(
                                fwd_duration, 0.0
                            )
                            bwd_config = dummy_config.copy()
                            bwd_config.split_mask_num = iter
                            configs_all[group_idx][_config_key(bwd_config)].stable_add(
                                0.0, bwd_duration
                            )

        if F.Dataflow.FetchOnDemand in local_dataflow_range:
            # Fetch-on-Demand. Tune whether to fuse.
            dummy_config.dataflow = F.Dataflow.FetchOnDemand
            FOD_fusion_range = (
                [pruned_FOD_fusion]
                if dataflow_prune and pruned_FOD_fusion is not None
                else [True, False]
            )
            for FOD_fusion in FOD_fusion_range:
                dummy_config.FOD_fusion = FOD_fusion
                set_group_config(model, names, dummy_config)
                inputs = clear_tensor_cache(inputs)
                fwd_duration, bwd_duration = torch_lattice_tune_timer(
                    model, inputs, tune_with_bwd
                )
                configs_all[group_idx][_config_key(dummy_config)].stable_add(
                    fwd_duration, bwd_duration
                )

        if F.Dataflow.GatherScatter in local_dataflow_range:
            # Gather-Scatter. Tune eps & mm_thresh
            dummy_config.dataflow = F.Dataflow.GatherScatter
            for epsilon in np.arange(0.0, 0.6, 0.1):
                for mm_thresh in [
                    0,
                    5000,
                    7500,
                    10000,
                    12500,
                    15000,
                    17500,
                    20000,
                    22500,
                    25000,
                ]:
                    dummy_config.epsilon, dummy_config.mm_thresh = epsilon, mm_thresh
                    set_group_config(model, names, dummy_config)
                    inputs = clear_tensor_cache(inputs)
                    fwd_duration, bwd_duration = torch_lattice_tune_timer(
                        model, inputs, tune_with_bwd
                    )
                    configs_all[group_idx][_config_key(dummy_config)].stable_add(
                        fwd_duration, bwd_duration
                    )


# @torch.no_grad()
def tune(
    model: nn.Module,
    data_loader: Iterable,
    n_samples: int = 100,
    collect_fn: Callable = lambda data: data,
    enable_fp16: bool = False,
    save_dir: str = ".torch-lattice-tune",
    tune_tag: str = "temp",
    force_retune: bool = False,
    dataflow_range: List = None,
    dataflow_prune: bool = False,
    tune_with_bwd: bool = False,
    verbose: bool = True,
    skip_warning: bool = False,
):
    """Tune sparse convolution backend configuration for a model.

    Args:
        model: Module to profile for convolution backend configuration.
        data_loader: Iterable that yields representative training samples.
        n_samples: Number of samples used while profiling candidate configs.
        collect_fn: Function that converts one data-loader item into model input.
            The tuned call is equivalent to ``model(collect_fn(data))`` unless the
            callable returns a structure consumed by the model itself.
        enable_fp16: Profile with half precision and CUDA autocast enabled.
        save_dir: Directory used to cache tuned configuration files.
        tune_tag: Cache file name under ``save_dir``.
        force_retune: Ignore an existing cache file and profile again.
        dataflow_range: Candidate convolution dataflows. When omitted, forward-only
            tuning checks implicit GEMM and Fetch-on-Demand; backward tuning uses
            implicit GEMM.
        dataflow_prune: Select the best dataflow before tuning lower-level config
            thresholds.
        tune_with_bwd: Include backward timing in the tuning objective.
        verbose: Print tuning progress and cache information.
        skip_warning: Suppress iterator and backend-mode warnings.
    """
    if dataflow_range is None:
        dataflow_range = (
            [F.Dataflow.ImplicitGEMM]
            if tune_with_bwd
            else [F.Dataflow.ImplicitGEMM, F.Dataflow.FetchOnDemand]
        )

    # An iterator can only be used once, so use with care.
    if isinstance(data_loader, Iterator):
        if not skip_warning:
            print(f"Warning: data_loader is an iterator of type {type(data_loader)}.")
            print("Take caution if data_loader is shared with other functions.")
    if not torch_lattice.backends.benchmark:  # type: ignore
        if not skip_warning:
            print(
                "Warning: to use tuning, "
                + "torch_lattice.backends.benchmark is automatically set to be true."
            )
        torch_lattice.backends.benchmark = True  # type: ignore

    dataflow_all: DefaultDict[
        Tuple[Any, ...], DefaultDict[Tuple[Any, ...], StableTimeAccumulator]
    ] = defaultdict(lambda: defaultdict(StableTimeAccumulator))
    configs_all: DefaultDict[
        Tuple[Any, ...], DefaultDict[Tuple[Any, ...], StableTimeAccumulator]
    ] = defaultdict(lambda: defaultdict(StableTimeAccumulator))
    name_to_group: DefaultDict[str, Tuple[Any, ...]] = {}
    group_to_name = defaultdict(list)
    device_id = int(str(next(model.parameters()).device).split(":")[-1])

    # hook function to store data for profiling
    # group the conv layers by stage
    def dump(module, inputs, outputs, name):
        if not module.transposed:
            tensor_stride = inputs[0].stride
        else:
            tensor_stride = tuple(
                inputs[0].stride[k] // make_ntuple(module.stride, ndim=3)[k]
                for k in range(3)
            )
        group_idx = (tensor_stride, module.kernel_size, module.stride, module.dilation)
        name_to_group[name] = group_idx
        group_to_name[group_idx].append(name)

    group_dataflow = {}
    group_configs = {}
    if (os.path.exists(os.path.join(save_dir, tune_tag))) and not force_retune:
        if verbose:
            print("Load existing tuned group configs")
        name_to_group, group_configs = torch.load(os.path.join(save_dir, tune_tag))
    else:
        handler_collection = []
        for name, module in model.named_modules():
            # register hook
            if isinstance(module, _TUNABLE_CONV_TYPES):
                if len(module.weight.data.shape) == 3:
                    _handler = module.register_forward_hook(
                        functools.partial(dump, name=name)
                    )
                    handler_collection.append(_handler)

        # Stage 0: Dump the model structure
        for i, feed_dict in enumerate(
            tqdm(
                data_loader,
                desc="Dump the model structure",
                leave=False,
                total=n_samples,
            )
        ):
            inputs = collect_fn(feed_dict)
            if enable_fp16:
                inputs = recursive_apply(inputs, lambda x: x.half())
                model = model.half()
            inputs = recursive_apply(inputs, lambda x: x.cuda())
            model = model.cuda()
            with torch.amp.autocast("cuda", enabled=enable_fp16):
                # generate dumps
                name_to_group = {}
                group_to_name = defaultdict(list)
                _ = model(inputs)
                # detach the hook
                for _handler in handler_collection:
                    _handler.remove()
            break

        # Stage 1: select best dataflow for each group (Prune the search space)
        if dataflow_prune:
            if len(dataflow_range) == 1:
                if verbose:
                    print(
                        f"Only 1 dataflow ({dataflow_range[0]}) is set. Skip dataflow selecting."
                    )
                dataflow_prune = False
            else:
                count = 0
                for i, feed_dict in enumerate(
                    tqdm(
                        data_loader,
                        desc="Select best dataflow for each group",
                        leave=False,
                        total=n_samples,
                    )
                ):
                    inputs = collect_fn(feed_dict)
                    if enable_fp16:
                        inputs = recursive_apply(inputs, lambda x: x.half())
                        model = model.half()
                    inputs = recursive_apply(inputs, lambda x: x.cuda())
                    model = model.cuda()
                    with torch.amp.autocast("cuda", enabled=enable_fp16):
                        if i == 0:
                            # device warm-up
                            for warm_iter in range(10):
                                _ = model(inputs)
                        dataflow_selector(
                            model,
                            inputs,
                            dataflow_range,
                            group_to_name,
                            dataflow_all,
                            tune_with_bwd,
                        )
                    count += 1
                    if count == n_samples:
                        break

                # Search for the best dataflow
                for group_idx in dataflow_all:
                    time_min = -1.0
                    for dataflow, FOD_fusion in dataflow_all[group_idx]:
                        if (
                            time_min < 0
                            or time_min
                            > dataflow_all[group_idx][
                                (dataflow, FOD_fusion)
                            ].get_total_time()
                        ):
                            time_min = dataflow_all[group_idx][
                                (dataflow, FOD_fusion)
                            ].get_total_time()
                            dataflow_best = dataflow
                            FOD_fusion_best = FOD_fusion
                    group_dataflow[group_idx] = {
                        "dataflow": dataflow_best,
                        "FOD_fusion": FOD_fusion_best,
                    }

        # Stage 2: Tune best configs for each group
        count = 0
        for i, feed_dict in enumerate(
            tqdm(
                data_loader,
                desc="Tuning best group configs",
                leave=False,
                total=n_samples,
            )
        ):
            inputs = collect_fn(feed_dict)
            if enable_fp16:
                inputs = recursive_apply(inputs, lambda x: x.half())
                model = model.half()
            inputs = recursive_apply(inputs, lambda x: x.cuda())
            model = model.cuda()
            with torch.amp.autocast("cuda", enabled=enable_fp16):
                if i == 0:
                    # device warm-up
                    for warm_iter in range(10):
                        _ = model(inputs)
                profile_model(
                    model,
                    inputs,
                    dataflow_range,
                    dataflow_prune,
                    group_to_name,
                    configs_all,
                    group_dataflow,
                    tune_with_bwd,
                )
            count += 1
            if count == n_samples:
                break

        # Search for the best configs for each group
        for group_idx in configs_all:
            time_min = -1.0
            for (
                ep,
                thresh,
                split_mask_num,
                split_mask_num_bwd,
                dataflow,
                ifsort,
                FOD_fusion,
                *rest,
            ) in configs_all[group_idx]:
                if len(rest) >= 2:
                    IGEMM_center_only = rest[0]
                    wgrad_split_k = rest[1]
                else:
                    IGEMM_center_only = False
                    wgrad_split_k = rest[0] if rest else 32
                if (
                    time_min < 0
                    or time_min
                    > configs_all[group_idx][
                        (
                            ep,
                            thresh,
                            split_mask_num,
                            split_mask_num_bwd,
                            dataflow,
                            ifsort,
                            FOD_fusion,
                            *rest,
                        )
                    ].get_total_time()
                ):
                    time_min = configs_all[group_idx][
                        (
                            ep,
                            thresh,
                            split_mask_num,
                            split_mask_num_bwd,
                            dataflow,
                            ifsort,
                            FOD_fusion,
                            *rest,
                        )
                    ].get_total_time()
                    ep_best = ep
                    thresh_best = thresh
                    split_mask_num_best = split_mask_num
                    split_mask_num_bwd_best = split_mask_num_bwd
                    dataflow_best = dataflow
                    ifsort_best = ifsort
                    FOD_fusion_best = FOD_fusion
                    wgrad_split_k_best = wgrad_split_k
                    IGEMM_center_only_best = IGEMM_center_only
            group_configs[group_idx] = {
                "epsilon": ep_best,
                "mm_thresh": thresh_best,
                "split_mask_num": split_mask_num_best,
                "split_mask_num_bwd": split_mask_num_bwd_best,
                "dataflow": dataflow_best,
                "ifsort": ifsort_best,
                "FOD_fusion": FOD_fusion_best,
                "wgrad_split_k": wgrad_split_k_best,
                "IGEMM_center_only": IGEMM_center_only_best,
            }

        # save tuned group configs
        if device_id == 0:
            if verbose:
                print("Save tuned group configs to", os.path.join(save_dir, tune_tag))
            os.makedirs(save_dir, exist_ok=True)
            torch.save((name_to_group, group_configs), os.path.join(save_dir, tune_tag))

    # modify the model
    for name, module in model.named_modules():
        if isinstance(module, _TUNABLE_CONV_TYPES):
            if name in name_to_group:
                layer_group_idx = name_to_group[name]
                if layer_group_idx in group_configs:
                    new_config = module._config
                    if new_config is None:
                        glb_config = F.conv_config.get_global_conv_config()
                        if glb_config is not None:
                            new_config = glb_config.copy()
                        else:
                            new_config = F.conv_config.get_default_conv_config().copy()
                    new_config.dataflow = group_configs[layer_group_idx]["dataflow"]
                    new_config.epsilon = group_configs[layer_group_idx]["epsilon"]
                    new_config.mm_thresh = group_configs[layer_group_idx]["mm_thresh"]
                    new_config.ifsort = group_configs[layer_group_idx]["ifsort"]
                    new_config.split_mask_num = group_configs[layer_group_idx][
                        "split_mask_num"
                    ]
                    new_config.split_mask_num_bwd = group_configs[layer_group_idx][
                        "split_mask_num_bwd"
                    ]
                    new_config.FOD_fusion = group_configs[layer_group_idx]["FOD_fusion"]
                    new_config.wgrad_split_k = group_configs[layer_group_idx].get(
                        "wgrad_split_k", 32
                    )
                    new_config.IGEMM_center_only = group_configs[layer_group_idx].get(
                        "IGEMM_center_only", False
                    )
                    module._config = new_config.copy()

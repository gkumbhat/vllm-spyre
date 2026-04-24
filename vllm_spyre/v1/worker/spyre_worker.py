"""A Spyre worker class."""

import contextlib
import functools
import json
import os
import platform
import signal
import sys
import threading
import time
import math
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Union, cast

import torch
import torch.distributed as dist
import vllm.envs as envs
from huggingface_hub import hf_hub_download
from vllm.config import VllmConfig
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.logger import init_logger
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

import vllm_spyre.envs as envs_spyre
import vllm_spyre.perf_metrics as perf_metrics
import vllm_spyre.utils as utils_spyre
from vllm_spyre.model_executor.model_loader import spyre_setup
from vllm_spyre.multimodal.mm_coordinator import MMCoordinator, cleanup_stale_shared_memory
from vllm_spyre.platform import SpyrePlatform
from vllm_spyre.v1.worker.spyre_model_runner import (
    ChunkedPrefillModelRunner,
    SpyrePoolingModelRunner,
    SupportedTask,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput


logger = init_logger(__name__)

_inside_warmup_mode = False


def new_request_data_builder(
    req_id: str,
    block_ids: tuple[list[int]],
    prompt_token_ids: list[int],
    sampling_params: SamplingParams | None,
    pooling_params: PoolingParams | None,
    prompt_embeds: torch.Tensor | None,
    mm_features: list | None,
) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        block_ids=block_ids,
        num_computed_tokens=0,
        lora_request=None,
        mm_features=mm_features or [],
        prompt_embeds=prompt_embeds,
    )


@contextlib.contextmanager
def _maybe_warmup_context(limit: int, world_size: int, rank: int):
    global _inside_warmup_mode
    warmup_context = contextlib.nullcontext
    if SpyrePlatform.is_backend_sendnn_enabled():
        from torch_sendnn import warmup_mode  # ty: ignore
        warmup_context = warmup_mode

    sendnn_exit = warmup_context.__exit__

    def __stagger_exit__(*args, **kwargs):
        with utils_spyre.stagger_region(limit, world_size, rank):
            sendnn_exit(*args, **kwargs)

    functools.update_wrapper(__stagger_exit__, sendnn_exit)
    warmup_context.__exit__ = __stagger_exit__  # type: ignore[method-assign]

    with warmup_context():
        _inside_warmup_mode = True
        yield
        _inside_warmup_mode = False


@contextlib.contextmanager
def use_torch_fx_backed_size_oblivious():
    from torch.fx.experimental import _config as config
    config.backed_size_oblivious = True  # ty: ignore[invalid-assignment]
    yield
    config.backed_size_oblivious = False


class SpyreWorker(WorkerBase):
    """A worker class that executes the model on a group of Spyre cores."""

    @property
    def is_pooling(self) -> bool:
        return self.model_config.runner_type == "pooling"

    @property
    def is_decoder(self) -> bool:
        return self.model_config.runner_type == "generate"

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def compile_or_warm_up_model(self) -> float:
        if self.is_decoder:
            return self._warmup_spyre_dynamic_size(self.restricted_tokens)
        if self.model_runner.is_multimodal:
            raise NotImplementedError("[WARMUP] multimodal models are not supported yet.")
        num_shape_combinations = len(self.spyre_warmup_shapes)
        logger.info(
            "[WARMUP] Starting for %d prompt/decode/batchsize-shape combinations...",
            len(self.spyre_warmup_shapes),
        )
        all_warmup_start_t = time.time()
        for i, (prompt_len, batch_size) in enumerate(
            [(s["prompt_length"], s["batch_size"]) for s in self.spyre_warmup_shapes]
        ):
            logger.info(
                "[WARMUP] (%d/%d) for prompt length %d with batch size %d...",
                i + 1, num_shape_combinations, prompt_len, batch_size,
            )
            self._warmup_spyre_fixed_size(prompt_len, self.restricted_tokens, batch_size)

        self.model_runner.complete_warmup()
        all_warmup_end_t = time.time()
        all_warmup_total_t = all_warmup_end_t - all_warmup_start_t
        self.perf_metrics.log("total warmup time", all_warmup_total_t)
        del self.perf_metrics
        logger.info(
            "[WARMUP] All %d combinations finished in %.3fs",
            num_shape_combinations, all_warmup_total_t,
        )
        return all_warmup_total_t

    def check_health(self) -> None:
        return

    def determine_available_memory(self) -> int:
        accurate_fake_kv_cache_size = (
            4 * self.model_config.max_model_len * self.scheduler_config.max_num_seqs
        )
        return 2 * accurate_fake_kv_cache_size

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        pass

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        self.redirect_logs_to_files()
        self.perf_metrics = perf_metrics.create_perf_metric_logger(rank)

        if self.parallel_config and is_driver_worker:
            assert rank % self.parallel_config.tensor_parallel_size == 0, (
                "Driver worker should be rank 0 of tensor parallel group."
            )

        self.model_runner: Union[ChunkedPrefillModelRunner, SpyrePoolingModelRunner]
        self.warmup_block_ids = 1

        if self.is_pooling:
            self.model_runner = SpyrePoolingModelRunner(
                self.vllm_config, self.is_driver_worker, self.rank
            )
            self.spyre_warmup_shapes = SpyrePlatform.get_warmup_shapes(
                self.vllm_config.scheduler_config
            )
        else:
            self.model_runner = ChunkedPrefillModelRunner(
                self.vllm_config, self.is_driver_worker, self.rank
            )

        self._env_initialized = False

        profiler_config = vllm_config.profiler_config
        if profiler_config.profiler == "torch":
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            self.profiler: TorchProfilerWrapper | None = TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU"],
            )
            if SpyrePlatform.is_backend_sendnn_enabled():
                logger.info_once(
                    "Traces will contain AIU events if PyTorch with "
                    "AIU profiling support is installed."
                )
                os.environ["ProfilerActivity"] = "PrivateUse1"  # noqa: SIM112
                dt_opt = os.environ.get("DT_OPT", "")
                options = dict(opt.split("=") for opt in dt_opt.split(",") if "=" in opt)
                if options.get("autopilot", "1") == "1":
                    logger.warning_once(
                        "autopilot on detected with profiling enabled. Add "
                        "autopilot=0 to DT_OPT to see individual AIU-kernel "
                        "execution in the trace."
                    )
        else:
            self.profiler = None

        # ── MMCoordinator ─────────────────────────────────────────────────────
        # Constructed here but NOT started. The poller thread starts only after
        # start_mm_coordinator_poller() is called via collective_rpc, which
        # connects to the Manager server started by SpyreExecutor.
        if self.parallel_config.world_size > 1 and not self.is_pooling:
            cleanup_stale_shared_memory()
            # Batching is used for MM encoding - requests are accumulated and
            # processed together in a single model forward pass.
            # num_encoder_threads is ignored (single encoder thread used)
            self.mm_coordinator: MMCoordinator | None = MMCoordinator(
                rank=self.rank,
                tp_size=self.parallel_config.world_size,
            )
            if isinstance(self.model_runner, ChunkedPrefillModelRunner):
                self.model_runner.set_mm_coordinator(self.mm_coordinator)
        else:
            self.mm_coordinator = None

    def init_distributed_environment(self) -> None:
        torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
        if SpyrePlatform.is_backend_sendnn_enabled():
            spyre_setup.spyre_dist_setup(
                rank=self.rank, world_size=self.parallel_config.world_size, verbose=True
            )
        torch.distributed.all_reduce(torch.zeros(1).cpu())

    def redirect_logs_to_files(self) -> None:
        if envs_spyre.VLLM_SPYRE_WORKER_LOG_REDIRECT_DIR:
            log_dir = Path(envs_spyre.VLLM_SPYRE_WORKER_LOG_REDIRECT_DIR)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"rank-{self.rank}.log"
            logger.warning("Redirecting all logs to %s", str(log_path))
            redirected_file = log_path.open("w")
            redirected_fd = redirected_file.fileno()
            os.dup2(redirected_fd, sys.stderr.fileno())
            os.dup2(redirected_fd, sys.stdout.fileno())

    def init_device(self) -> None:
        if platform.machine() == "s390x":
            from torch.serialization import LoadEndianness
            torch.serialization.set_default_load_endianness(LoadEndianness.LITTLE)

        if not self._env_initialized:
            init_distributed_environment(
                world_size=self.parallel_config.world_size,
                rank=self.rank,
                distributed_init_method="env://",
                backend="gloo",
                timeout=timedelta(minutes=envs_spyre.VLLM_SPYRE_GLOO_TIMEOUT_MINUTES),
            )
            if self.parallel_config.world_size > 1:
                self.init_distributed_environment()
            elif SpyrePlatform.is_backend_sendnn_enabled():
                spyre_setup.spyre_setup()

            ensure_model_parallel_initialized(
                self.parallel_config.tensor_parallel_size,
                self.parallel_config.pipeline_parallel_size,
            )
            self._env_initialized = True

        set_random_seed(self.model_config.seed)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        assert self._env_initialized

        is_local = os.path.isdir(self.model_config.model)
        if is_local:
            cf_file = os.path.join(self.model_config.model, "config.json")
        else:
            cf_file = hf_hub_download(
                repo_id=self.model_config.model,
                revision=self.model_config.revision,
                filename="config.json",
            )
        with open(cf_file, "rb") as f:
            config = json.load(f)

        restricted_tokens = []
        if tok := config.get("bos_token_id") is not None:
            restricted_tokens.append(int(tok))
        if tok := config.get("eos_token_id") is not None:
            restricted_tokens.append(int(tok))
        self.restricted_tokens = restricted_tokens

        logger.info("load model...")
        load_model_start_t = time.time()
        self.model_runner.load_model()
        load_model_total_t = time.time() - load_model_start_t
        self.perf_metrics.log("load model time", load_model_total_t, model=self.model_config.model)
        logger.info("load model took %.3fs", load_model_total_t)

    def _gen_warmup_block_ids(self, num_tokens: int) -> tuple[list[int]]:
        num_blocks = math.ceil(num_tokens / 64)
        start = self.warmup_block_ids
        end = start + num_blocks
        self.warmup_block_ids = end
        return ([i for i in range(start, end)],)

    def _warmup_spyre_dynamic_size(self, special_token_ids) -> float:
        warmup_start_t = time.time()
        model_runner: ChunkedPrefillModelRunner = cast(ChunkedPrefillModelRunner, self.model_runner)
        vocab_size = model_runner.vocab_size
        valid_token_ids = [i for i in range(1, vocab_size) if i not in set(special_token_ids)]
        valid_token_ids_tensor = torch.tensor(
            valid_token_ids, dtype=torch.long, device=torch.device("cpu")
        )
        num_decode_tokens = 2
        req_count = 3 if self.model_config.quantization is not None else 2

        mm_model_utils = self.model_runner.get_mm_utils()
        if mm_model_utils:
            mm_warmup_inputs = mm_model_utils.get_warmup_inputs(req_count)
            warmup_tokens = mm_warmup_inputs.input_ids
            warmup_embeds_tensor = mm_warmup_inputs.input_embeds
            mm_features = mm_warmup_inputs.mm_features
            prompt_len = len(warmup_tokens[0])
        else:
            prompt_len = 42
            warmup_tokens_tensor = valid_token_ids_tensor[
                torch.randint(0, len(valid_token_ids_tensor), (3, prompt_len))
            ]
            warmup_tokens = [wt.tolist() for wt in warmup_tokens_tensor]
            warmup_embeds_tensor = [None] * req_count
            mm_features = None

        requests = [
            new_request_data_builder(
                req_id="warmup-%d" % (i),
                prompt_token_ids=warmup_tokens[i],
                block_ids=self._gen_warmup_block_ids(len(warmup_tokens[i])),
                sampling_params=SamplingParams(max_tokens=num_decode_tokens),
                pooling_params=None,
                prompt_embeds=warmup_embeds_tensor[i],
                mm_features=mm_features,
            )
            for i in range(req_count)
        ]

        warmup_requests = requests[:-1]
        deploy_req = requests[-1]
        model_runner.pre_warmup()

        with _maybe_warmup_context(
            envs_spyre.VLLM_SPYRE_MAX_LOAD_PROCESSES, self.parallel_config.world_size, self.rank
        ):
            self._dynamic_warmup(
                requests=warmup_requests,
                prompt_len=prompt_len,
                valid_token_ids_tensor=valid_token_ids_tensor,
            )

        deploy_req.sampling_params = SamplingParams(
            temperature=1.0, top_k=10, top_p=0.9, min_p=0.9,
            presence_penalty=0.5, frequency_penalty=0.5,
            repetition_penalty=1.2, max_tokens=4, min_tokens=1, logprobs=1,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[deploy_req],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={deploy_req.req_id: prompt_len},
            total_num_scheduled_tokens=prompt_len,
            finished_req_ids=set(),
            **_get_extra_args(),
        )
        logger.info("[WARMUP] Deploying to device...")
        self.execute_model(scheduler_output)
        self._cleanup_model_runner(request=[deploy_req])
        model_runner.complete_warmup()

        warmup_total_t = time.time() - warmup_start_t
        compile_cache_str = (
            "enabled" if int(os.getenv("TORCH_SENDNN_CACHE_ENABLE", "0")) else "disabled"
        )
        logger.info(
            "[WARMUP] Finished in %.3fs (compilation cache %s)",
            warmup_total_t, compile_cache_str,
        )
        maybe_override_signals_handler()
        return warmup_total_t

    def _cleanup_model_runner(self, request) -> None:
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            finished_req_ids=set([r.req_id for r in request]),
            **_get_extra_args(),
        )
        self.execute_model(scheduler_output)
        model_runner: ChunkedPrefillModelRunner = cast(ChunkedPrefillModelRunner, self.model_runner)
        model_runner.tkv = 0

    def _warmup_spyre_fixed_size(self, prompt_len, special_token_ids, batch_size):
        assert self.is_pooling, "only pooling models have fixed warmup shapes"
        warmup_start_t = time.time()
        vocab_size = self.model_runner.vocab_size
        valid_token_ids = [i for i in range(1, vocab_size) if i not in set(special_token_ids)]
        valid_token_ids_tensor = torch.tensor(
            valid_token_ids, dtype=torch.long, device=torch.device("cpu")
        )
        warmup_tokens_tensor = valid_token_ids_tensor[
            torch.randint(0, len(valid_token_ids_tensor), (batch_size, prompt_len))
        ]
        pooling_params = PoolingParams(task="embed")
        dummy_requests = [
            new_request_data_builder(
                req_id=f"warmup_{i}",
                prompt_token_ids=warmup_tokens_tensor[i].tolist(),
                block_ids=self._gen_warmup_block_ids(len(warmup_tokens_tensor[i])),
                sampling_params=None,
                pooling_params=pooling_params,
                prompt_embeds=None,
                mm_features=None,
            )
            for i in range(batch_size)
        ]
        req_ids, new_token_ids, new_block_ids, num_computed_tokens = [], [], [], []
        for req in dummy_requests:
            req_ids.append(req.req_id)
            new_token_ids.append(
                [valid_token_ids_tensor[torch.randint(0, len(valid_token_ids_tensor), (1,)).item()]]  # ty: ignore
            )
            new_block_ids.append([req.block_ids])
            num_computed_tokens.append(req.num_computed_tokens)

        cached_request_data = CachedRequestData.make_empty()
        cached_request_data.req_ids = req_ids
        cached_request_data.new_block_ids = new_block_ids
        cached_request_data.new_token_ids = new_token_ids
        cached_request_data.num_computed_tokens = num_computed_tokens

        for r in dummy_requests:
            assert r.prompt_token_ids is not None
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=dummy_requests,
            scheduled_cached_reqs=cached_request_data,
            num_scheduled_tokens={r.req_id: self._get_num_tokens(r) for r in dummy_requests},
            total_num_scheduled_tokens=sum(prompt_len for _ in range(batch_size)),
            finished_req_ids=set(),
            **_get_extra_args(),
        )
        logger.info("[WARMUP] Compiling graphs...")
        with _maybe_warmup_context(
            envs_spyre.VLLM_SPYRE_MAX_LOAD_PROCESSES, self.parallel_config.world_size, self.rank
        ):
            self._warmup_model_forward_pass(scheduler_output, dummy_requests, cached_request_data)
        self.perf_metrics.log(
            "warmup 1 time", time.time() - warmup_start_t,
            batch_size=batch_size, prompt_len=prompt_len,
        )
        logger.info("[WARMUP] Deploying to device...")
        warmup2_start_t = time.time()
        self._warmup_model_forward_pass(scheduler_output, dummy_requests, cached_request_data)
        self.perf_metrics.log(
            "warmup 2 time", time.time() - warmup2_start_t,
            batch_size=batch_size, prompt_len=prompt_len,
        )
        compile_cache_str = (
            "enabled" if int(os.getenv("TORCH_SENDNN_CACHE_ENABLE", "0")) else "disabled"
        )
        logger.info(
            "[WARMUP] Prompt length %d finished in %.3fs (compilation cache %s)",
            prompt_len, time.time() - warmup_start_t, compile_cache_str,
        )
        maybe_override_signals_handler()

    @use_torch_fx_backed_size_oblivious()
    def _dynamic_warmup(
        self,
        requests: list[NewRequestData],
        prompt_len: int,
        valid_token_ids_tensor: torch.Tensor,
    ) -> None:
        assert _inside_warmup_mode, "must be inside warmup context"
        req_count = len(requests)
        for idx, req in enumerate(requests):
            scheduler_output = SchedulerOutput(
                scheduled_new_reqs=[req],
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                num_scheduled_tokens={req.req_id: prompt_len},
                total_num_scheduled_tokens=prompt_len,
                finished_req_ids=set(),
                **_get_extra_args(),
            )
            logger.info("[WARMUP] Prefill [%s/%s]...", idx + 1, req_count)
            self.execute_model(scheduler_output)

        random_token_id = lambda: torch.randint(0, len(valid_token_ids_tensor), (1,)).item()
        cached_request_data = CachedRequestData.make_empty()
        cached_request_data.req_ids = [req.req_id for req in requests]
        cached_request_data.new_block_ids = []
        for req in requests:
            if len(req.prompt_token_ids) % 64 == 0:
                cached_request_data.new_block_ids.append(self._gen_warmup_block_ids(1))
            else:
                cached_request_data.new_block_ids.append(([],))
        cached_request_data.new_token_ids = [
            [valid_token_ids_tensor[random_token_id()]] for _ in requests
        ]
        cached_request_data.num_computed_tokens = [prompt_len for _ in requests]

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_request_data,
            num_scheduled_tokens={req.req_id: 1 for req in requests},
            total_num_scheduled_tokens=1,
            finished_req_ids=set(),
            **_get_extra_args(),
        )
        logger.info("[WARMUP] Decode...")
        self.execute_model(scheduler_output)
        self._cleanup_model_runner(request=requests)

    def _warmup_model_forward_pass(
        self,
        scheduler_output: SchedulerOutput,
        requests: list[NewRequestData],
        cached_request_data: CachedRequestData,
    ):
        assert self.is_pooling, "only pooling models have fixed warmup shapes"
        scheduler_output.scheduled_new_reqs = requests
        scheduler_output.scheduled_cached_reqs = CachedRequestData.make_empty()
        scheduler_output.num_scheduled_tokens = {
            r.req_id: self._get_num_tokens(r) for r in requests
        }
        self.execute_model(scheduler_output)

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Please set --profiler-config to enable it."
            )
        if is_start:
            self.profiler.start()
        else:
            if self.profiler is None:
                logger.warning("Profiler was not started, nothing to stop.")
                return
            self.profiler.stop()

    @property
    def do_metadata_broadcast(self) -> bool:
        return True

    @property
    def kv_cache(self) -> list[list[torch.Tensor]] | None:
        return None

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def sample_tokens(self, grammar_output: "GrammarOutput | None") -> ModelRunnerOutput:
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
        return EMPTY_MODEL_RUNNER_OUTPUT

    # ── MMCoordinator RPC methods ─────────────────────────────────────────────

    def start_mm_coordinator_poller(self) -> None:
        """Connect to Manager server and start the poller thread.

        Called once per worker via collective_rpc("start_mm_coordinator_poller")
        with NO arguments. Each worker independently connects to the Manager
        server started by SpyreExecutor, fetches its Queue proxy by rank, and
        starts its daemon poller thread.

        Works under both fork and spawn — the Manager server is reachable via
        loopback socket regardless of how the worker process was started.
        """
        if self.mm_coordinator is not None:
            self.mm_coordinator.start_poller()

    def shutdown_mm_coordinator(self) -> None:
        """Gracefully shut down the MMCoordinator on this worker."""
        if self.mm_coordinator is not None:
            try:
                self.mm_coordinator.shutdown()
            except Exception as e:
                logger.debug("Rank %d: MMCoordinator shutdown error: %s", self.rank, e)

    # ── MM encoding submission ────────────────────────────────────────────────

    def _maybe_submit_mm_encoding(self, scheduler_output: SchedulerOutput) -> None:
        """Submit vision encoding for new MM requests at top of execute_model.

        Rank-0 only. Submitting here — before model_runner.execute_model() —
        gives the encoder thread maximum lead time. The model runner's input
        preparation work runs concurrently with vision encoding. When
        get_embedding() is called inside _prepare_chunked_prefill, the future
        may already be done, giving a zero-wait return.

        Idempotent: submit_encoding() is a no-op for already-submitted requests.
        During warmup (poller not started), encoding runs synchronously.
        """
        if self.rank != 0 or self.mm_coordinator is None:
            return

        for req in scheduler_output.scheduled_new_reqs:
            mm_features = getattr(req, "mm_features", None)
            if not mm_features:
                continue

            if self.mm_coordinator._fms_model is None:
                self.mm_coordinator.set_model_and_utils(
                    self.model_runner.model.fms_model,
                    self.model_runner.model.mm_model_utils,
                )

            full_input_tokens = torch.tensor(
                req.prompt_token_ids, dtype=torch.int64
            ).unsqueeze(0)

            # During warmup, run encoding synchronously to avoid batching
            # issues with synthetic warmup requests that share mm_features
            # Check: (1) warmup context flag, (2) poller not started, or (3) request ID indicates warmup
            if (_inside_warmup_mode or self.mm_coordinator._my_queue is None
                or req.req_id.startswith("warmup-")):
                # Warmup path: encode synchronously and broadcast via shared memory
                # so non-zero ranks can receive the embedding via get_embedding()
                embedding = self.model_runner.model.mm_model_utils.get_maybe_mm_embeddings(
                    self.model_runner.model.fms_model,
                    full_input_tokens,
                    mm_features,
                    is_decode=False,
                )
                # Store in shared memory and broadcast to all ranks (same as normal path)
                self.mm_coordinator._store_and_broadcast(req.req_id, embedding)
                # Also cache locally for rank-0 fast path
                with self.mm_coordinator._lock:
                    self.mm_coordinator._local_embeddings[req.req_id] = embedding
                logger.debug("Rank 0: Warmup MM encoding done for %s", req.req_id)
            else:
                # Normal path: submit to background batch processor
                self.mm_coordinator.submit_encoding(
                    request_id=req.req_id,
                    input_ids=full_input_tokens,
                    mm_features=mm_features,
                )

    # ── Main execution entry point ────────────────────────────────────────────

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        if self.profiler is not None:
            self.profiler.step()

        # Submit MM encoding at the very top, before model_runner.execute_model.
        # This gives the background encoder thread maximum lead time and allows
        # encoding to overlap with _prepare_chunked_prefill's input preparation.
        # All TP ranks proceed into the forward pass immediately after — no
        # extra RPC overhead, no cross-process tensor serialization.
        self._maybe_submit_mm_encoding(scheduler_output)

        output = self.model_runner.execute_model(scheduler_output)

        # Release shared memory for completed requests (rank-0 only)
        if (
            self.mm_coordinator is not None
            and self.rank == 0
            and scheduler_output.finished_req_ids
        ):
            for req_id in scheduler_output.finished_req_ids:
                self.mm_coordinator.release(req_id)

        return output if self.is_driver_worker else None

    def _get_num_tokens(self, r: NewRequestData) -> int:
        assert r.prompt_token_ids is not None, "requests should have tokens!"
        return len(r.prompt_token_ids)


def maybe_override_signals_handler():
    if not (envs.VLLM_ENABLE_V1_MULTIPROCESSING and envs_spyre.VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER):
        return

    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def _get_extra_args() -> dict:
    return {
        "free_encoder_mm_hashes": [],
        "scheduled_spec_decode_tokens": {},
        "scheduled_encoder_inputs": {},
        "num_common_prefix_blocks": [],
    }
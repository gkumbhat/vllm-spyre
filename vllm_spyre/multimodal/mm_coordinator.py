"""Multimodal Coordinator for deduplicating vision encoder computation across TP workers.

Queue sharing model (works under both fork AND spawn)
-----------------------------------------------------
We use a standard multiprocessing.Manager with a shared dict to hold queues.

  1. SpyreExecutor starts a Manager server in the EngineCore process on a
     known loopback port before workers are spawned.
  2. Each worker independently connects to the Manager server in
     start_poller() and retrieves its Queue from the shared dict.
  3. collective_rpc("start_mm_coordinator_poller") sends NO arguments —
     workers find the Manager server themselves via the known port.

The Queue objects are Manager.Queue() objects (created by multiprocessing.Manager).
Workers hold proxy references. The put/get calls go through the Manager socket —
this is slower than a direct Queue, but it is correct under both fork and spawn,
and is only on the MM result notification path (not the model forward pass hot path).

To minimize Manager socket overhead on the put side, rank-0's encoder thread
calls put() on tp_size queues once per request. Workers' poller threads call
get() once per request. Neither is on the forward-pass critical path.
"""

import concurrent.futures
import hashlib
import multiprocessing
import multiprocessing.managers
import os
import re
import threading
import time
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _PendingEncoding:
    """Holds a pending MM encoding request."""
    request_id: str
    input_ids: torch.Tensor
    mm_features: list
    future: concurrent.futures.Future | None = None

# ---------------------------------------------------------------------------
# Manager server
# ---------------------------------------------------------------------------

_MANAGER_HOST = "127.0.0.1"
_MANAGER_PORT = int(os.environ.get("SPYRE_MM_COORD_PORT", "15799"))
_MANAGER_AUTHKEY = b"spyre-mm-coord-v1"


# Global dict that will hold queues - accessed via Manager
# This dict lives in the Manager server process
_server_queues: dict = {}  # type: ignore[misc]


def _manager_server_init(tp_size: int) -> None:
    """Initializer run in the Manager server process.

    Creates tp_size multiprocessing.Queue objects and stores them in the global dict.
    """
    global _server_queues
    _server_queues = {}
    for rank in range(tp_size):
        _server_queues[rank] = multiprocessing.Queue()
    logger.info("Manager server initialized for tp_size=%d, created %d queues", tp_size, len(_server_queues))


def _get_queue(rank: int):
    """Return the Queue for the given rank. Called via Manager proxy."""
    global _server_queues
    if rank not in _server_queues:
        # Create on-demand if not pre-created (shouldn't happen in normal flow)
        _server_queues[rank] = multiprocessing.Queue()
    return _server_queues[rank]


class _MMQueueManager(multiprocessing.managers.BaseManager):
    """Manager server that holds proxy-able Queue objects for each TP rank."""
    pass


# Register methods - multiprocessing.Queue() returned from callable gets proxied
_MMQueueManager.register("get_queue", callable=_get_queue)


def start_manager_server(tp_size: int) -> _MMQueueManager:
    """Start the Manager server in a separate process with queues accessible.

    Must be called BEFORE workers are spawned so the server is ready when
    workers try to connect. The returned manager instance must be kept alive
    for the lifetime of the workers.

    Args:
        tp_size: Number of TP workers (queues will be created internally).

    Returns:
        The running manager instance. Caller must hold a reference.
    """
    manager = _MMQueueManager(
        address=(_MANAGER_HOST, _MANAGER_PORT),
        authkey=_MANAGER_AUTHKEY,
    )
    manager.start(initializer=_manager_server_init, initargs=(tp_size,))
    logger.info(
        "MMCoordinator: Manager server started on %s:%d with %d queues",
        _MANAGER_HOST, _MANAGER_PORT, tp_size
    )
    return manager


def connect_to_manager(rank: int, tp_size: int, timeout: float = 30.0) -> multiprocessing.Queue:
    """Connect to the Manager server and return this rank's Queue proxy.

    Called from workers inside start_poller(). Retries for up to `timeout`
    seconds in case the server hasn't started yet.

    Args:
        rank: This worker's TP rank.
        tp_size: Expected number of queues (for validation).
        timeout: Seconds to retry before giving up.

    Returns:
        A proxy to the Queue for this rank.

    Raises:
        RuntimeError: If the server is not reachable within timeout.
    """
    client = _MMQueueManager(
        address=(_MANAGER_HOST, _MANAGER_PORT),
        authkey=_MANAGER_AUTHKEY,
    )
    deadline = time.monotonic() + timeout
    attempt = 0
    while True:
        try:
            client.connect()
            break
        except (ConnectionRefusedError, OSError):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Rank {rank}: Could not connect to MMCoordinator Manager "
                    f"server at {_MANAGER_HOST}:{_MANAGER_PORT} after {timeout}s. "
                    "Ensure SpyreExecutor started the server before spawning workers."
                )
            attempt += 1
            wait = min(0.5 * attempt, 2.0)
            logger.debug(
                "Rank %d: Manager not ready yet, retrying in %.1fs (attempt %d)",
                rank, wait, attempt,
            )
            time.sleep(wait)

    # Get this rank's queue proxy from the Manager server
    queue_proxy = client.get_queue(rank)
    logger.debug(
        "Rank %d: Connected to Manager server, got queue proxy", rank
    )
    return queue_proxy


# ---------------------------------------------------------------------------
# Shared memory helpers
# ---------------------------------------------------------------------------

_SHM_PREFIX = "spyre_mm_"


def _make_shm_name(request_id: str) -> str:
    """Derive a POSIX-safe shared memory name from an arbitrary request_id."""
    digest = hashlib.md5(request_id.encode(), usedforsecurity=False).hexdigest()[:16]
    return f"{_SHM_PREFIX}{digest}"


def cleanup_stale_shared_memory() -> None:
    """Unlink stale shared memory segments from crashed runs.

    Safe to call concurrently from multiple workers.
    """
    shm_dir = "/dev/shm"
    try:
        for name in os.listdir(shm_dir):
            if re.match(r"spyre_mm_[0-9a-f]{16}$", name):
                try:
                    shm = SharedMemory(name=name)
                    shm.unlink()
                    logger.debug("Cleaned up stale shared memory: %s", name)
                except Exception:
                    pass
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# MMCoordinator
# ---------------------------------------------------------------------------

class MMCoordinator:
    """Coordinates multimodal embedding computation across tensor-parallel workers.

    One instance per worker process. Rank-0 encodes; all ranks consume.

    Lifecycle:
        1. SpyreExecutor calls start_manager_server(queues) before spawning
           workers. Manager server starts on a known loopback port.
        2. Workers are spawned. MMCoordinator(rank, tp_size) constructed in
           each worker's __init__. Poller NOT started yet.
        3. SpyreExecutor calls collective_rpc("start_mm_coordinator_poller")
           with NO args. Each worker connects to the Manager server,
           retrieves its Queue proxy, and starts the poller daemon thread.
        4. Rank-0: set_model_and_utils() called once after model load.
        5. Top of execute_model(): rank-0 calls submit_encoding() for new
           MM requests (idempotent — no-op if already submitted).
        6. In _prepare_chunked_prefill(): all ranks call get_embedding().
           Rank-0 also calls wait_for_encoding() first.
        7. On request completion: rank-0 calls release().
        8. On shutdown: all workers call shutdown().
    """

    def __init__(self, rank: int, tp_size: int, num_encoder_threads: int = 2) -> None:
        self.rank = rank
        self.tp_size = tp_size

        # ── rank-0 only ──────────────────────────────────────────────────────
        self._encoder_pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._encoding_futures: dict[str, concurrent.futures.Future] = {}
        self._shm_registry: dict[str, tuple[SharedMemory, tuple, torch.dtype]] = {}
        self._fms_model: torch.nn.Module | None = None
        self._mm_model_utils: Any = None
        self._model_ref_lock = threading.Lock()

        # Batching support: queue of pending requests to batch together
        self._pending_encodings: list[_PendingEncoding] = []
        self._batch_lock = threading.Lock()
        self._batch_ready = threading.Event()
        # Batch window: accumulate requests for up to 5ms before processing
        # This allows multiple concurrent requests to be batched together
        self._batch_window_ms = float(os.environ.get("VLLM_SPYRE_MM_BATCH_WINDOW_MS", "5"))

        if rank == 0:
            # Single encoder thread - batching happens on this thread
            self._encoder_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="mm-encoder",
            )
            # Start the batch processor thread
            self._batch_processor = threading.Thread(
                target=self._batch_processor_loop,
                daemon=True,
                name="mm-batch-processor",
            )
            self._batch_processor.start()

        # ── all ranks ────────────────────────────────────────────────────────
        # Queue proxy set by start_poller(); None until then
        self._my_queue: Any = None
        # All queues needed by rank-0 to broadcast results; set by start_poller()
        self._all_queues: list[Any] = []

        self._local_events: dict[str, threading.Event] = {}
        self._local_embeddings: dict[str, torch.Tensor] = {}
        self._local_errors: dict[str, Exception] = {}

        self._lock = threading.Lock()
        self._shutdown = False
        self._poller: threading.Thread | None = None
        self._batch_processor: threading.Thread | None = None

    # ── Poller startup ────────────────────────────────────────────────────────

    def start_poller(self) -> None:
        """Connect to Manager server, get queue proxy, start poller thread.

        Called once per worker via collective_rpc("start_mm_coordinator_poller").
        Works under both fork and spawn — connects to the Manager server
        independently rather than relying on inherited state.

        Raises:
            RuntimeError: If called twice, or if Manager server is unreachable.
        """
        if self._poller is not None:
            raise RuntimeError(
                f"MMCoordinator.start_poller() called twice on rank {self.rank}"
            )

        # All ranks get their own queue proxy by connecting to the Manager.
        # Rank-0 additionally gets all queues so it can broadcast results.
        self._my_queue = connect_to_manager(self.rank, self.tp_size)

        if self.rank == 0:
            # Rank-0 needs proxies for all queues to broadcast results.
            # Create a new client connection to get all queues.
            # The queues are cached in the Manager server process by rank,
            # so all clients see the same queue instances.
            client = _MMQueueManager(
                address=(_MANAGER_HOST, _MANAGER_PORT),
                authkey=_MANAGER_AUTHKEY,
            )
            client.connect()
            self._all_queues = [client.get_queue(r) for r in range(self.tp_size)]
            logger.info(
                "Rank 0: Got %d queue proxies for broadcast: %s",
                len(self._all_queues), [type(q).__name__ for q in self._all_queues]
            )
        else:
            self._all_queues = []

        self._poller = threading.Thread(
            target=self._poll_results,
            daemon=True,
            name=f"mm-poller-rank{self.rank}",
        )
        self._poller.start()
        logger.debug("Rank %d: MMCoordinator poller started", self.rank)

    # ── Model reference (rank-0 only) ─────────────────────────────────────────

    def set_model_and_utils(
        self,
        fms_model: torch.nn.Module,
        mm_model_utils: Any,
    ) -> None:
        """Set model references for encoding. Safe to call multiple times."""
        with self._model_ref_lock:
            self._fms_model = fms_model
            self._mm_model_utils = mm_model_utils
        logger.debug("Rank 0: MMCoordinator model references updated")

    # ── Encoding submission (rank-0 only) ────────────────────────────────────

    def submit_encoding(
        self,
        request_id: str,
        input_ids: torch.Tensor,
        mm_features: list,
    ) -> None:
        """Submit vision encoding to the background batch queue. Rank-0 only.

        Idempotent: safe to call on every execute_model() iteration.
        Check + submit happen under a single lock acquisition (prevents TOCTOU).

        Requests are queued and processed in batches by the encoder thread.
        """
        assert self.rank == 0, "submit_encoding must only be called on rank-0"
        assert self._encoder_pool is not None

        if self._my_queue is None:
            raise RuntimeError(
                "Rank 0: submit_encoding() called before start_poller(). "
                "Ensure collective_rpc('start_mm_coordinator_poller') ran."
            )

        with self._model_ref_lock:
            fms_model = self._fms_model
            mm_model_utils = self._mm_model_utils

        if fms_model is None or mm_model_utils is None:
            raise RuntimeError(
                "Rank 0: set_model_and_utils() must be called before submit_encoding()"
            )

        # Single lock: check + submit atomically (prevents TOCTOU race)
        with self._lock:
            if request_id in self._encoding_futures:
                return
            # Create a future for this request
            future: concurrent.futures.Future = concurrent.futures.Future()
            self._encoding_futures[request_id] = future
            # Queue the request for batched processing
            pending = _PendingEncoding(
                request_id=request_id,
                input_ids=input_ids,
                mm_features=mm_features,
                future=future,
            )
            with self._batch_lock:
                self._pending_encodings.append(pending)
            # Signal that there's work to do
            self._batch_ready.set()

        logger.debug("Rank 0: Queued MM encoding for request %s", request_id)

    def _process_batch(self) -> None:
        """Process all pending encodings as a single batch.

        Called by the encoder thread. Batches requests with the same
        mm_features structure (same image sizes) for efficiency.
        """
        with self._batch_lock:
            pending = self._pending_encodings.copy()
            self._pending_encodings.clear()
        self._batch_ready.clear()

        if not pending:
            return

        with self._model_ref_lock:
            fms_model = self._fms_model
            mm_model_utils = self._mm_model_utils

        if fms_model is None or mm_model_utils is None:
            # Model not ready - fail all pending requests
            for p in pending:
                p.future.set_exception(
                    RuntimeError("Model not ready for MM encoding")
                )
            return

        # Group requests by image size for batching
        # Requests with different image sizes need separate batches
        batches: dict[tuple, list[_PendingEncoding]] = {}
        for p in pending:
            # Extract image size key from mm_features
            img_size_key = "default"
            if p.mm_features:
                spec = p.mm_features[0].data if p.mm_features[0].data else {}
                if "image_sizes" in spec:
                    img_sizes = spec["image_sizes"].data
                    if isinstance(img_sizes, torch.Tensor):
                        img_size_key = tuple(img_sizes.tolist())
                    else:
                        img_size_key = str(img_sizes)
            batches.setdefault(img_size_key, []).append(p)

        # Process each batch
        for batch_key, batch_requests in batches.items():
            self._encode_batch(batch_requests, fms_model, mm_model_utils)

    def _encode_batch(
        self,
        pending_requests: list[_PendingEncoding],
        fms_model: torch.nn.Module,
        mm_model_utils: Any,
    ) -> None:
        """Encode a batch of requests together.

        Stacks input_ids and mm_features for true batched encoding
        in a single model forward pass.
        """
        try:
            t_start = time.perf_counter()

            # Batch multiple requests together
            # Stack input_ids: [1, seq] -> [batch, seq]
            batch_input_ids = torch.cat([p.input_ids for p in pending_requests], dim=0)
            batch_size = len(pending_requests)
            seq_lens = [p.input_ids.shape[1] for p in pending_requests]

            # Collect all mm_features for batched processing
            batch_mm_features = []
            for p in pending_requests:
                if p.mm_features:
                    batch_mm_features.extend(p.mm_features)

            # Run batched encoding - single forward pass for all images
            # Returns shape: [batch*seq, dim] (flattened)
            batch_embeddings = mm_model_utils.get_maybe_mm_embeddings(
                fms_model, batch_input_ids, batch_mm_features, is_decode=False
            )

            # Split embeddings back to individual requests based on sequence lengths
            # batch_embeddings shape: [total_seq, dim] where total_seq = sum(seq_lens)
            embed_dim = batch_embeddings.shape[-1]
            offset = 0
            for i, p in enumerate(pending_requests):
                seq_len = seq_lens[i]
                embedding = batch_embeddings[offset : offset + seq_len]  # [seq_i, dim]
                self._store_and_broadcast(p.request_id, embedding)
                p.future.set_result(None)
                offset += seq_len

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.info(
                "[MM] Batch encoding complete for %d requests, time: %.2fms",
                len(pending_requests), elapsed_ms,
            )

        except Exception as e:
            logger.exception("Rank 0: Batch MM encoding failed: %s", e)
            for p in pending_requests:
                if not p.future.done():
                    p.future.set_exception(e)

    def _store_and_broadcast(self, request_id: str, embedding: torch.Tensor) -> None:
        """Store embedding in shared memory and broadcast to all ranks."""
        shm_name = _make_shm_name(request_id)
        try:
            shm = SharedMemory(create=True, size=embedding.nbytes, name=shm_name)
        except FileExistsError:
            logger.warning("Rank 0: shm %s exists, unlinking and retrying", shm_name)
            try:
                stale = SharedMemory(name=shm_name)
                stale.unlink()
                stale.close()
            except Exception:
                pass
            shm = SharedMemory(create=True, size=embedding.nbytes, name=shm_name)

        buf = torch.frombuffer(shm.buf, dtype=embedding.dtype)
        buf.copy_(embedding.flatten())

        with self._lock:
            self._shm_registry[request_id] = (shm, embedding.shape, embedding.dtype)

        metadata = (
            request_id,
            shm_name,
            tuple(embedding.shape),
            str(embedding.dtype),
        )
        self._broadcast_to_queues(metadata)

    def _encode(
        self,
        request_id: str,
        input_ids: torch.Tensor,
        mm_features: list,
        fms_model: torch.nn.Module,
        mm_model_utils: Any,
    ) -> None:
        """Run vision encoding and broadcast result to all rank queues.

        On success: writes embedding to shared memory, broadcasts metadata.
        On failure: broadcasts error sentinel so no rank hangs.

        fms_model and mm_model_utils are passed as arguments so we do not
        hold _model_ref_lock for the full encoding duration.
        """
        try:
            t_start = time.perf_counter()
            embedding = mm_model_utils.get_maybe_mm_embeddings(
                fms_model, input_ids, mm_features, is_decode=False
            )
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.info(
                "[MM] Encoding complete for request %s, shape %s, time: %.2fms",
                request_id, embedding.shape, elapsed_ms,
            )

            shm_name = _make_shm_name(request_id)
            try:
                shm = SharedMemory(create=True, size=embedding.nbytes, name=shm_name)
            except FileExistsError:
                logger.warning("Rank 0: shm %s exists, unlinking and retrying", shm_name)
                try:
                    stale = SharedMemory(name=shm_name)
                    stale.unlink()
                    stale.close()
                except Exception:
                    pass
                shm = SharedMemory(create=True, size=embedding.nbytes, name=shm_name)

            buf = torch.frombuffer(shm.buf, dtype=embedding.dtype)
            buf.copy_(embedding.flatten())

            with self._lock:
                self._shm_registry[request_id] = (shm, embedding.shape, embedding.dtype)

            metadata = (
                request_id,
                shm_name,
                tuple(embedding.shape),
                str(embedding.dtype),
            )
            self._broadcast_to_queues(metadata)

        except Exception as e:
            logger.exception(
                "Rank 0: MM encoding failed for request %s: %s", request_id, e
            )
            # Broadcast error sentinel so all waiting ranks unblock
            self._broadcast_to_queues((request_id, None, None, None))
            raise

    def _broadcast_to_queues(self, item: tuple) -> None:
        """Put item into every rank's queue proxy. Best-effort."""
        logger.debug(
            "Rank 0: Broadcasting to %d queues, item=%s",
            len(self._all_queues), item[0]
        )
        for i, q in enumerate(self._all_queues):
            try:
                q.put(item)
                logger.debug("Rank 0: Put item in queue %d", i)
            except Exception as e:
                logger.error(
                    "Rank 0: Failed to enqueue result for rank %d: %s", i, e
                )

    def wait_for_encoding(
        self, request_id: str, timeout: float | None = 30.0
    ) -> None:
        """Block until encoding is done. Rank-0 only. No-op on other ranks."""
        if self.rank != 0:
            return

        with self._lock:
            future = self._encoding_futures.get(request_id)

        if future is None:
            logger.warning(
                "Rank 0: wait_for_encoding() — no future for %s", request_id
            )
            return

        try:
            future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"Rank 0: Timed out waiting for MM encoding of "
                f"request {request_id} after {timeout}s"
            )

    def release(self, request_id: str) -> None:
        """Unlink shared memory for a completed request. Rank-0 only."""
        if self.rank != 0:
            return

        with self._lock:
            entry = self._shm_registry.pop(request_id, None)
            self._encoding_futures.pop(request_id, None)

        if entry is not None:
            shm, _, _ = entry
            try:
                shm.close()
                shm.unlink()
                logger.debug("Rank 0: Released shared memory for %s", request_id)
            except Exception as e:
                logger.warning(
                    "Rank 0: Error releasing shared memory for %s: %s",
                    request_id, e,
                )

    def _batch_processor_loop(self) -> None:
        """Background thread that processes batches of MM encodings.

        Waits for requests to accumulate within a batch window, then processes
        them together in a single model forward pass for efficiency.
        """
        logger.debug("Rank 0: Batch processor thread started (window=%dms)",
                     int(self._batch_window_ms))

        while not self._shutdown:
            # Wait for work signal
            signaled = self._batch_ready.wait(timeout=0.1)

            if self._shutdown:
                break

            if not signaled:
                # Timeout - check if there's pending work
                with self._batch_lock:
                    if not self._pending_encodings:
                        continue

            # Small additional wait to allow more requests to accumulate
            # This improves batching efficiency for near-simultaneous requests
            time.sleep(self._batch_window_ms / 1000.0)

            # Process all pending requests as a batch
            try:
                self._process_batch()
            except Exception as e:
                logger.exception("Rank 0: Batch processor error: %s", e)

        logger.debug("Rank 0: Batch processor thread shutting down")

    # ── Poller thread (all ranks) ─────────────────────────────────────────────

    def _poll_results(self) -> None:
        """Drain this rank's queue and populate the local embedding cache.

        Items: (request_id, shm_name, shape, dtype_str)
        - None → shutdown sentinel
        - shm_name is None → encoding failed on rank-0
        - Normal → read from shared memory, set local event
        """
        logger.debug("Rank %d: MM result poller running", self.rank)

        while not self._shutdown:
            try:
                item = self._my_queue.get(timeout=0.5)
            except Exception as e:
                # Manager proxy raises various exceptions for empty/timeout;
                # these are expected when no MM encoding is in progress.
                err_str = str(e).lower()
                if "empty" in err_str or "timed out" in err_str or "timeout" in err_str:
                    # Expected: no items in queue, keep polling
                    continue
                if self._shutdown:
                    break
                # Real error - log with full details and keep polling
                logger.warning(
                    "Rank %d: Queue get error (type: %s): %s",
                    self.rank, type(e).__name__, e,
                )
                continue

            consecutive_errors = 0

            if item is None:
                logger.debug("Rank %d: Received shutdown sentinel", self.rank)
                break

            req_id, shm_name, shape, dtype_str = item
            logger.info(
                "Rank %d: Poller received item for request %s",
                self.rank, req_id
            )

            if shm_name is None:
                # Encoding failed on rank-0
                logger.error(
                    "Rank %d: Encoding error sentinel for request %s",
                    self.rank, req_id,
                )
                error = RuntimeError(
                    f"Vision encoding failed on rank-0 for request {req_id}. "
                    "Check rank-0 logs for the root cause."
                )
                with self._lock:
                    self._local_errors[req_id] = error
                    if req_id in self._local_events:
                        self._local_events[req_id].set()
                continue

            try:
                t_start = time.perf_counter()
                dtype = getattr(torch, dtype_str.replace("torch.", ""))
                shm = SharedMemory(name=shm_name)
                tensor = torch.frombuffer(shm.buf, dtype=dtype).reshape(shape).clone()
                shm.close()
                elapsed_ms = (time.perf_counter() - t_start) * 1000

                logger.info(
                    "[MM] Request %s: embedding ready on rank %d (shm read: %.2fms)",
                    req_id, self.rank, elapsed_ms,
                )

                with self._lock:
                    self._local_embeddings[req_id] = tensor
                    if req_id in self._local_events:
                        self._local_events[req_id].set()

            except Exception as e:
                logger.exception(
                    "Rank %d: Failed to read shm for request %s: %s",
                    self.rank, req_id, e,
                )
                with self._lock:
                    self._local_errors[req_id] = e
                    if req_id in self._local_events:
                        self._local_events[req_id].set()

    # ── Embedding retrieval (all ranks) ───────────────────────────────────────

    def get_embedding(
        self,
        request_id: str,
        timeout: float | None = 30.0,
    ) -> torch.Tensor:
        """Return embedding for request_id, waiting if not yet available.

        Rank-0 fast path: checks future directly first.
        All ranks: event-wait path as fallback.
        """
        # ── Rank-0 fast path ──────────────────────────────────────────────────
        if self.rank == 0:
            with self._lock:
                if request_id in self._local_embeddings:
                    return self._local_embeddings.pop(request_id)
                future = self._encoding_futures.get(request_id)

            if future is not None and future.done():
                exc = future.exception()
                if exc is not None:
                    raise exc
            # Fall through to event-wait

        # ── All ranks: event-wait ─────────────────────────────────────────────
        with self._lock:
            if request_id in self._local_embeddings:
                return self._local_embeddings.pop(request_id)
            if request_id in self._local_errors:
                raise self._local_errors.pop(request_id)
            event = threading.Event()
            self._local_events[request_id] = event

        t_start = time.perf_counter()
        succeeded = event.wait(timeout=timeout)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        if not succeeded:
            with self._lock:
                self._local_events.pop(request_id, None)
            raise TimeoutError(
                f"Rank {self.rank}: Timed out waiting for MM embedding for "
                f"request {request_id} after {timeout}s. "
                "Check rank-0 encoder thread logs."
            )

        with self._lock:
            self._local_events.pop(request_id, None)
            if request_id in self._local_errors:
                raise self._local_errors.pop(request_id)
            if request_id in self._local_embeddings:
                logger.info(
                    "[MM] Request %s embedding consumed on rank %d "
                    "(total wait: %.2fms)",
                    request_id, self.rank, elapsed_ms,
                )
                return self._local_embeddings.pop(request_id)

        raise RuntimeError(
            f"Rank {self.rank}: Event set for request {request_id} but "
            "no embedding or error found. This is a bug in MMCoordinator."
        )

    def maybe_get_embedding(self, request_id: str) -> torch.Tensor | None:
        """Non-blocking check. Returns embedding if cached, None otherwise."""
        with self._lock:
            return self._local_embeddings.pop(request_id, None)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Gracefully shut down this rank's coordinator."""
        self._shutdown = True

        if self._my_queue is not None:
            try:
                self._my_queue.put(None)  # wake the poller
            except Exception as e:
                logger.debug("Rank %d: Error sending shutdown sentinel: %s", self.rank, e)

        if self._poller is not None:
            self._poller.join(timeout=2.0)
            if self._poller.is_alive():
                logger.warning("Rank %d: Poller thread did not stop in 2s", self.rank)

        if self.rank == 0:
            # Wake up batch processor if waiting
            self._batch_ready.set()

            if self._batch_processor is not None:
                self._batch_processor.join(timeout=2.0)
                if self._batch_processor.is_alive():
                    logger.warning("Rank 0: Batch processor thread did not stop in 2s")

            if self._encoder_pool is not None:
                self._encoder_pool.shutdown(wait=True, cancel_futures=True)

            with self._lock:
                for req_id, (shm, _, _) in list(self._shm_registry.items()):
                    try:
                        shm.close()
                        shm.unlink()
                    except Exception:
                        pass
                self._shm_registry.clear()
                self._encoding_futures.clear()
                self._pending_encodings.clear()

        logger.debug("Rank %d: MMCoordinator shutdown complete", self.rank)
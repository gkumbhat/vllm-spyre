"""SpyreExecutor - Custom executor for vllm-spyre with MMCoordinator support.

Queue sharing strategy
----------------------
plain multiprocessing.Queue cannot be pickled and therefore cannot be sent
through collective_rpc (which serializes args via pickle). Fork inheritance
works only when workers are forked — but vllm may use spawn (fresh Python
interpreter per worker), in which case no module-level state is inherited.

We use a multiprocessing.managers.BaseManager server:

  1. SpyreExecutor starts the Manager server in the EngineCore process
     BEFORE workers are spawned, with a known loopback address.
  2. Workers are spawned (fork or spawn — both work).
  3. collective_rpc("start_mm_coordinator_poller") sends NO arguments.
     Each worker independently connects to the Manager server, fetches its
     Queue proxy by rank, and starts its poller thread.

Queue proxy objects returned by the Manager ARE picklable (they contain only
host, port, and auth token), so workers can receive them from the Manager
without going through collective_rpc.

Hot-path impact
---------------
The Manager socket overhead is only on the MM result notification path
(one put() per request from rank-0's encoder thread, one get() per request
from each rank's poller thread). It is NOT on the forward-pass critical path.
"""

from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING

from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.logger import init_logger

from vllm_spyre.multimodal.mm_coordinator import start_manager_server

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)


class SpyreExecutor(MultiprocExecutor):
    """Executor for Spyre with MMCoordinator support.

    Extends MultiprocExecutor to start the MMCoordinator Manager server and
    distribute the signal to start poller threads in each worker.

    Lifecycle:
        __init__ (before super):
            Start Manager server with tp_size queues
            (works before or after spawn — server binds on a loopback port)

        super().__init__():
            Spawn worker subprocesses (fork or spawn)
            Workers run init_device, load_model, compile_or_warm_up_model

        __init__ (after super):
            collective_rpc("start_mm_coordinator_poller") — no args
            Each worker connects to Manager, gets its Queue proxy, starts poller

        execute_model:
            Delegates to super() — no MM-specific logic in executor

        shutdown:
            collective_rpc("shutdown_mm_coordinator")
            super().shutdown()
            Manager server shuts down
    """

    def __init__(self, vllm_config, monitor_workers: bool = True) -> None:
        tp_size = vllm_config.parallel_config.world_size

        # ── Step 1: Start Manager server before workers are spawned ───────────
        # The server binds on a loopback port. Workers connect to it after
        # spawn by calling connect_to_manager() in start_poller(). We keep a
        # reference to the manager instance so the server stays alive.
        if tp_size > 1:
            # start_manager_server creates Manager.Queue objects internally
            # and returns the manager instance. We hold self._mm_manager to keep
            # the server alive for the lifetime of the executor.
            self._mm_manager = start_manager_server(tp_size)
            self._has_mm_coordinator = True
            logger.debug(
                "SpyreExecutor: MMCoordinator Manager server started (%d queues)",
                tp_size,
            )
        else:
            self._mm_manager = None
            self._has_mm_coordinator = False

        # ── Step 2: Spawn workers ─────────────────────────────────────────────
        # Workers are spawned here. The Manager server is already listening
        # so workers can connect in start_poller() once the signal arrives.
        super().__init__(vllm_config, monitor_workers)

        # ── Step 3: Signal workers to start their poller threads ──────────────
        # collective_rpc with no args — safe to pickle (empty tuple).
        # Each worker connects to the Manager server independently and starts
        # its poller daemon thread.
        if self._has_mm_coordinator:
            try:
                self.collective_rpc("start_mm_coordinator_poller")
                logger.info(
                    "SpyreExecutor: MMCoordinator pollers started on %d workers",
                    tp_size,
                )
            except Exception:
                logger.exception(
                    "SpyreExecutor: Failed to start MMCoordinator pollers"
                )
                raise

    # ── Hot path: no override ─────────────────────────────────────────────────
    # execute_model is NOT overridden. MM encoding submission happens inside
    # SpyreWorker._maybe_submit_mm_encoding() at the top of the worker's own
    # execute_model() — no RPC overhead, no cross-process tensor serialization.

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Shut down MMCoordinator on all workers, then the Manager server."""
        if self._has_mm_coordinator:
            try:
                self.collective_rpc("shutdown_mm_coordinator", timeout=10)
            except Exception as e:
                logger.debug(
                    "SpyreExecutor: MMCoordinator shutdown error (non-fatal): %s", e
                )

        super().shutdown()

        if self._mm_manager is not None:
            try:
                self._mm_manager.shutdown()
                logger.debug("SpyreExecutor: Manager server shut down")
            except Exception as e:
                logger.debug("SpyreExecutor: Manager shutdown error: %s", e)
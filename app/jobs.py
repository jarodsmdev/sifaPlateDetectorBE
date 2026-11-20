"""
Cola acotada de trabajos de procesamiento de patentes.

Cada trabajo se ejecuta en un hilo aparte (para no bloquear el event loop) y
recibe un evento de cancelación que ``process_image_pipeline`` revisa entre
etapas. El número de trabajadores y el largo de la cola son explícitos.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field

from app.services import PipelineCancelled, process_image_pipeline

logger = logging.getLogger("sifa.jobs")


class QueueFullError(Exception):
    """La cola de trabajos no tiene espacio."""


@dataclass
class PlateJob:
    img: object
    cancel: threading.Event = field(default_factory=threading.Event)
    future: asyncio.Future | None = None


def _silence_future(future: asyncio.Future) -> None:
    # Evita el aviso "Future exception was never retrieved" si el cliente ya
    # se fue por timeout y nadie espera el resultado.
    if not future.cancelled():
        future.exception()


class PlateJobRunner:
    def __init__(self, max_concurrency: int, max_queue: int):
        self._max_concurrency = max(1, max_concurrency)
        self._max_queue = max(1, max_queue)
        self._queue: asyncio.Queue | None = None
        self._workers: list[asyncio.Task] = []

    async def start(self) -> None:
        if self._queue is not None:
            return
        self._queue = asyncio.Queue(maxsize=self._max_queue)
        self._workers = [
            asyncio.create_task(self._worker())
            for _ in range(self._max_concurrency)
        ]

    async def stop(self) -> None:
        if self._queue is None:
            return
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._queue = None
        self._workers = []

    async def submit(self, img) -> PlateJob:
        if self._queue is None:
            raise RuntimeError("PlateJobRunner no está iniciado")
        if self._queue.full():
            raise QueueFullError()
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(_silence_future)
        job = PlateJob(img=img, future=future)
        self._queue.put_nowait(job)
        return job

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job is None:
                    return
                if job.cancel.is_set():
                    if not job.future.done():
                        job.future.set_result(None)
                    continue
                try:
                    result = await asyncio.to_thread(
                        process_image_pipeline, job.img, job.cancel
                    )
                except PipelineCancelled:
                    if not job.future.done():
                        job.future.set_result(None)
                except Exception as exc:
                    if not job.future.done():
                        job.future.set_exception(exc)
                else:
                    if not job.future.done():
                        job.future.set_result(result)
            finally:
                self._queue.task_done()

"""Pruebas del pool acotado y de la cancelación cooperativa del pipeline."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from app.jobs import PlateJobRunner, QueueFullError
from app.services import PipelineCancelled, process_image_pipeline

IMG = np.zeros((10, 10, 3), dtype=np.uint8)


def test_runner_procesa_y_devuelve_resultado():
    async def scenario():
        runner = PlateJobRunner(max_concurrency=1, max_queue=1)
        await runner.start()
        try:
            with patch("app.jobs.process_image_pipeline", return_value=["ok"]):
                job = await runner.submit(IMG)
                return await asyncio.wait_for(job.future, timeout=2)
        finally:
            await runner.stop()

    assert asyncio.run(scenario()) == ["ok"]


def test_runner_rechaza_cuando_la_cola_esta_llena():
    async def scenario():
        runner = PlateJobRunner(max_concurrency=1, max_queue=1)
        await runner.start()
        started = threading.Event()
        release = threading.Event()

        def fake_pipeline(img, cancel):
            started.set()
            release.wait(timeout=5)
            return []

        try:
            with patch("app.jobs.process_image_pipeline", side_effect=fake_pipeline):
                job1 = await runner.submit(IMG)
                await asyncio.to_thread(started.wait, 2)
                job2 = await runner.submit(IMG)
                with pytest.raises(QueueFullError):
                    await runner.submit(IMG)

                release.set()
                await asyncio.wait_for(job1.future, timeout=2)
                await asyncio.wait_for(job2.future, timeout=2)
        finally:
            release.set()
            await runner.stop()

    asyncio.run(scenario())


def test_pipeline_no_arranca_si_ya_esta_cancelado():
    cancel = threading.Event()
    cancel.set()

    with patch("app.services.detect_plate") as detect:
        with pytest.raises(PipelineCancelled):
            process_image_pipeline(IMG, cancel)
        detect.assert_not_called()


def test_pipeline_se_detiene_antes_del_ocr_si_cancelan():
    cancel = threading.Event()
    det = SimpleNamespace(
        bounding_box=SimpleNamespace(x1=0, y1=0, x2=5, y2=5),
        confidence=0.9,
    )

    def fake_detect(_img):
        cancel.set()
        return [det]

    with patch("app.services.detect_plate", side_effect=fake_detect), \
            patch("app.services.read_plate") as read:
        with pytest.raises(PipelineCancelled):
            process_image_pipeline(IMG, cancel)
        read.assert_not_called()

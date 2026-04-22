from __future__ import annotations

import multiprocessing as mp
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from polygon_widget.application.processing import ContourExtractionSettings, DisplaySettings, SaveOptions
from polygon_widget.batch_processor import _batch_process_worker


class BatchProcessWorkerTests(unittest.TestCase):
    def test_batch_worker_drains_queue_and_returns_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_paths: list[str] = []
            for index in range(2):
                image = np.zeros((32, 32), dtype=np.uint8)
                image[8:24, 8:24] = 255
                image_path = str(Path(temporary_directory) / f"sample-{index}.png")
                self.assertTrue(cv2.imwrite(image_path, image))
                image_paths.append(image_path)

            context = mp.get_context("spawn")
            file_queue = context.Queue()
            result_queue = context.Queue()
            cancel_event = context.Event()
            for image_path in image_paths:
                file_queue.put(image_path)

            process = context.Process(
                target=_batch_process_worker,
                args=(
                    file_queue,
                    result_queue,
                    cancel_event,
                    {"steps": []},
                    ContourExtractionSettings(min_area=1.0),
                    None,
                    SaveOptions(),
                    DisplaySettings(),
                    "en",
                ),
            )
            process.start()
            process.join(timeout=20)

            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 0)

            messages = []
            while not result_queue.empty():
                messages.append(result_queue.get())

            results = [message[1] for message in messages if message[0] == "result"]
            progress = [message for message in messages if message[0] == "progress"]
            done = [message for message in messages if message[0] == "worker_done"]

            self.assertEqual({result.image_path for result in results}, set(image_paths))
            self.assertTrue(all(len(result.polygons) == 1 for result in results))
            self.assertEqual(len(progress), len(image_paths))
            self.assertEqual(len(done), 1)


if __name__ == "__main__":
    mp.freeze_support()
    unittest.main()

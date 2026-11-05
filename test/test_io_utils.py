# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Tests for io_utils extensionless video file handling (D99228861)."""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from sam3.model.io_utils import load_video_frames


class TestLoadVideoFramesRouting(unittest.TestCase):
    """Test that load_video_frames routes paths correctly based on extension."""

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_mp4_extension_routes_to_video_loader(
        self, mock_load_video: MagicMock
    ) -> None:
        """Paths with .mp4 extension should route to load_video_frames_from_video_file."""
        mock_load_video.return_value = ("frames", 480, 640)
        result = load_video_frames(
            video_path="/tmp/test_video.mp4",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()
        self.assertEqual(result, ("frames", 480, 640))

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_mov_extension_routes_to_video_loader(
        self, mock_load_video: MagicMock
    ) -> None:
        """Paths with .mov extension should route to load_video_frames_from_video_file."""
        mock_load_video.return_value = ("frames", 480, 640)
        load_video_frames(
            video_path="/tmp/test_video.mov",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_extensionless_oil_path_routes_to_video_loader(
        self, mock_load_video: MagicMock
    ) -> None:
        """Extensionless OIL paths should attempt video loading (D99228861 fix)."""
        mock_load_video.return_value = ("frames", 480, 640)
        result = load_video_frames(
            video_path="oil://fb_permanent/abc123def456",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()
        self.assertEqual(result, ("frames", 480, 640))

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_extensionless_bare_hash_routes_to_video_loader(
        self, mock_load_video: MagicMock
    ) -> None:
        """Bare hash paths without extension should attempt video loading."""
        mock_load_video.return_value = ("frames", 480, 640)
        result = load_video_frames(
            video_path="/data/videos/abc123def456",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()
        self.assertEqual(result, ("frames", 480, 640))

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_extensionless_path_raises_on_decode_failure(
        self, mock_load_video: MagicMock
    ) -> None:
        """Extensionless path that fails to decode should raise NotImplementedError."""
        mock_load_video.side_effect = RuntimeError("Could not decode video")
        with self.assertRaises(NotImplementedError) as ctx:
            load_video_frames(
                video_path="oil://fb_permanent/corrupted_file",
                image_size=256,
                offload_video_to_cpu=True,
            )
        self.assertIn("failed to load", str(ctx.exception))
        self.assertIn("oil://fb_permanent/corrupted_file", str(ctx.exception))

    @patch("sam3.model.io_utils.load_video_frames_from_image_folder")
    def test_directory_routes_to_image_folder_loader(
        self, mock_load_folder: MagicMock
    ) -> None:
        """Directory paths should route to load_video_frames_from_image_folder."""
        mock_load_folder.return_value = ("frames", 480, 640)
        with tempfile.TemporaryDirectory() as tmpdir:
            load_video_frames(
                video_path=tmpdir,
                image_size=256,
                offload_video_to_cpu=True,
            )
            mock_load_folder.assert_called_once()

    def test_dummy_video_pattern(self) -> None:
        """<load-dummy-video-N> pattern should return dummy frames."""
        frames, h, w = load_video_frames(
            video_path="<load-dummy-video-5>",
            image_size=64,
            offload_video_to_cpu=True,
        )
        self.assertEqual(frames.shape[0], 5)  # 5 frames
        self.assertEqual(h, 480)
        self.assertEqual(w, 640)

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_unknown_extension_routes_to_video_loader(
        self, mock_load_video: MagicMock
    ) -> None:
        """Paths with unrecognized extensions should attempt video loading."""
        mock_load_video.return_value = ("frames", 480, 640)
        result = load_video_frames(
            video_path="/tmp/video.xyz",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()
        self.assertEqual(result, ("frames", 480, 640))


class TestDiskBackedVideoFrames(unittest.TestCase):
    def test_black_white_scalar_batch_slice_ownership_and_close(self):
        import numpy as np
        import torch
        from sam3.model.io_utils import DiskBackedVideoFrames

        class Capture:
            def __init__(self):
                self.frames = iter([np.zeros((2, 2, 3), dtype=np.uint8),
                                    np.full((2, 2, 3), 255, dtype=np.uint8)])
                self.released = False

            def read(self):
                frame = next(self.frames, None)
                return frame is not None, frame

            def release(self):
                self.released = True

        cap = Capture()
        carrier = DiskBackedVideoFrames(cap, 2, (0.5,) * 3, (0.5,) * 3)
        carrier.device = torch.device("cpu")
        try:
            self.assertEqual(len(carrier), 2)
            self.assertTrue(cap.released)
            self.assertEqual(float(carrier[1].min()), 1.0, "white RGB must normalize to +1, not +509")
            self.assertEqual(carrier.dtype, torch.float16)
            black = torch.full((3, 2, 2), -1.0, dtype=torch.float16)
            white = torch.ones((3, 2, 2), dtype=torch.float16)
            scalar = carrier[0]
            batch = carrier[torch.tensor([0])]
            sliced = carrier[0:1]
            torch.testing.assert_close(scalar, black, rtol=0, atol=0)
            torch.testing.assert_close(carrier[1], white, rtol=0, atol=0)
            torch.testing.assert_close(batch, black.unsqueeze(0), rtol=0, atol=0)
            torch.testing.assert_close(sliced, black.unsqueeze(0), rtol=0, atol=0)
            scalar.fill_(0)
            # Each read owns its tensor; writes cannot mutate the spool or peers.
            torch.testing.assert_close(batch, black.unsqueeze(0), rtol=0, atol=0)
            torch.testing.assert_close(sliced, black.unsqueeze(0), rtol=0, atol=0)
            torch.testing.assert_close(carrier[0], black, rtol=0, atol=0)
            batch.fill_(1)
            torch.testing.assert_close(sliced, black.unsqueeze(0), rtol=0, atol=0)
            with self.assertRaises(IndexError):
                carrier[2]
        finally:
            carrier.close()
        self.assertTrue(carrier._file.closed)
        with self.assertRaises(ValueError):
            carrier[0]

    def test_real_cv2_frames_match_torchcodec_transform_without_resizing(self):
        import cv2
        import torch
        from sam3.model.io_utils import (
            AsyncVideoFileLoaderWithTorchCodec,
            load_video_frames_from_video_file_using_cv2,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_lossless_fixture(directory)
            capture = cv2.VideoCapture(str(path))
            carrier, height, width = load_video_frames_from_video_file_using_cv2(
                str(path), 8, offload_video_to_cpu=True
            )
            carrier.device = torch.device("cpu")
            # Exercise the real preprocessing method on the SAME decoded pixels;
            # decoder initialization/background allocation is a separate concern.
            transform = AsyncVideoFileLoaderWithTorchCodec.__new__(AsyncVideoFileLoaderWithTorchCodec)
            transform.image_size = 8
            transform.img_mean = torch.full((3, 1, 1), 0.5, dtype=torch.float16)
            transform.img_std = torch.full((3, 1, 1), 0.5, dtype=torch.float16)
            transform.offload_video_to_cpu = True
            try:
                self.assertEqual((height, width), (8, 8))
                for index in range(len(carrier)):
                    ok, frame = capture.read()
                    self.assertTrue(ok, f"cv2 missing fixture frame {index}")
                    decoded_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    decoded_tensor = torch.from_numpy(decoded_rgb).permute(2, 0, 1)
                    before = decoded_tensor.clone()
                    expected = transform._transform_frame(decoded_tensor)
                    torch.testing.assert_close(carrier[index], expected, rtol=0, atol=0)
                    torch.testing.assert_close(decoded_tensor, before, rtol=0, atol=0)
            finally:
                capture.release()
                carrier.close()

    def test_real_lossless_torchcodec_and_cv2_decoder_pixels_agree(self):
        import cv2
        import numpy as np
        from sam3.model.io_utils import TorchCodecDecoder

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_lossless_fixture(directory)
            decoder = TorchCodecDecoder(str(path), device="cpu", num_threads=1)
            capture = cv2.VideoCapture(str(path))
            try:
                self.assertEqual(len(decoder), 3)
                for index in range(len(decoder)):
                    ok, frame = capture.read()
                    self.assertTrue(ok, f"cv2 missing fixture frame {index}")
                    expected = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    decoded = decoder[index].permute(1, 2, 0).numpy()
                    difference = np.abs(expected.astype(np.int16) - decoded.astype(np.int16))
                    self.assertEqual(int(difference.max()), 0,
                                     f"actual decoder RGB mismatch at frame {index}, separate from normalization")
            finally:
                capture.release()

    def test_native_eager_cv2_black_white_fp16_arithmetic_on_cpu(self):
        import torch
        from sam3.model.io_utils import load_video_frames_from_video_file_using_cv2

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_lossless_fixture(directory)
            # Only device transfer is replaced for this CPU arithmetic check.
            # Real cv2 decoding and real Torch casts/division/normalization run.
            # This deliberately provides no CUDA/device-execution evidence.
            with patch.object(torch.Tensor, "cuda", lambda tensor: tensor):
                frames, height, width = load_video_frames_from_video_file_using_cv2(
                    str(path), 8, offload_video_to_cpu=False
                )
            self.assertEqual((height, width), (8, 8))
            self.assertEqual(float(frames[1].max()), 1.0, "white RGB must normalize to +1, not +509")
            self.assertEqual(frames.dtype, torch.float16)
            torch.testing.assert_close(frames[0], torch.full((3, 8, 8), -1.0, dtype=torch.float16), rtol=0, atol=0)
            torch.testing.assert_close(frames[1], torch.ones((3, 8, 8), dtype=torch.float16), rtol=0, atol=0)

    @staticmethod
    def _write_lossless_fixture(directory):
        from pathlib import Path
        import cv2
        import numpy as np

        path = Path(directory) / "rgb-ffv1.mkv"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 5.0, (8, 8))
        if not writer.isOpened():
            raise RuntimeError("the real pixel regression requires OpenCV/FFmpeg FFV1 encoding")
        try:
            writer.write(np.zeros((8, 8, 3), dtype=np.uint8))
            writer.write(np.full((8, 8, 3), 255, dtype=np.uint8))
            writer.write(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))
        finally:
            writer.release()
        return path

    def test_decode_failure_closes_capture_and_spool(self):
        from sam3.model.io_utils import DiskBackedVideoFrames
        class Capture:
            released = False
            def read(self):
                raise RuntimeError("decode failed")
            def release(self):
                self.released = True
        cap = Capture()
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            DiskBackedVideoFrames(cap, 2, (0.5,)*3, (0.5,)*3)
        self.assertTrue(cap.released)

    def test_start_session_accepts_actual_multiplex_init_signature(self):
        from sam3.model.sam3_base_predictor import Sam3BasePredictor
        class MultiplexSignature:
            def init_state(self, resource_path, offload_video_to_cpu=False,
                           async_loading_frames=False, use_torchcodec=False,
                           use_cv2=False, input_is_mp4=False):
                return {"resource":resource_path, "offload":offload_video_to_cpu}
        predictor = Sam3BasePredictor()
        predictor.model = MultiplexSignature()
        predictor.async_loading_frames = True
        response = predictor.handle_request({"type":"start_session", "resource_path":"video.mp4",
                                              "offload_video_to_cpu":True})
        self.assertEqual(predictor._get_session(response["session_id"])["state"],
                         {"resource":"video.mp4", "offload":True})
        predictor.handle_request({"type":"close_session", "session_id":response["session_id"],
                                  "run_gc_collect":False})

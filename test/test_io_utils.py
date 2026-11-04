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
    def test_native_pixels_scalar_batch_indexing_and_close(self):
        import cv2
        import numpy as np
        import torch
        from sam3.model.io_utils import DiskBackedVideoFrames

        source = [np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3),
                  np.full((3, 4, 3), 251, dtype=np.uint8)]
        class Capture:
            def __init__(self):
                self.frames = iter(source)
                self.released = False
            def read(self):
                frame = next(self.frames, None)
                return frame is not None, frame
            def release(self):
                self.released = True
        cap = Capture()
        carrier = DiskBackedVideoFrames(cap, 2, (0.5,)*3, (0.5,)*3)
        carrier.device = torch.device("cpu")
        try:
            self.assertEqual(len(carrier), 2)
            self.assertTrue(cap.released)
            expected_rgb = cv2.resize(cv2.cvtColor(source[0], cv2.COLOR_BGR2RGB),
                                      (2,2), interpolation=cv2.INTER_CUBIC)
            expected = (torch.from_numpy(expected_rgb.astype(np.float32)).permute(2,0,1)-0.5)/0.5
            torch.testing.assert_close(carrier[0], expected, rtol=0, atol=0)
            torch.testing.assert_close(carrier[torch.tensor([0])], expected.unsqueeze(0), rtol=0, atol=0)
            torch.testing.assert_close(carrier[0:1], expected.unsqueeze(0), rtol=0, atol=0)
            with self.assertRaises(IndexError):
                carrier[2]
        finally:
            carrier.close()
        self.assertTrue(carrier._file.closed)
        with self.assertRaises(ValueError):
            carrier[0]

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

import importlib.util
import sys
import types
import unittest

import numpy as np


def _install_missing_module_stub(name: str):
    if importlib.util.find_spec(name) is None:
        sys.modules[name] = types.ModuleType(name)


_install_missing_module_stub("sounddevice")
_install_missing_module_stub("torch")
_install_missing_module_stub("torchaudio")

from src.audio.capture import compute_audio_level  # noqa: E402


class AudioLevelTests(unittest.TestCase):
    def test_empty_audio_is_silence(self):
        self.assertEqual(compute_audio_level(np.array([], dtype=np.float32)), -60.0)

    def test_non_finite_audio_is_silence(self):
        audio = np.array([np.nan, np.inf], dtype=np.float32)
        self.assertEqual(compute_audio_level(audio), -60.0)

    def test_regular_audio_level_is_finite(self):
        audio = np.array([0.5, -0.5], dtype=np.float32)
        self.assertAlmostEqual(compute_audio_level(audio), -6.0206, places=3)


if __name__ == "__main__":
    unittest.main()

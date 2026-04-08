import json
import threading
from pathlib import Path

from src.logging_handler import logger

DEFAULT_CONFIG = {
    "audio_device_index": None,
    "audio_channel": 0,
    "sample_rate": 16000,
    "silence_duration": 0.8,
    "min_chunk_duration": 1.5,
    "max_chunk_duration": 12.0,
    "context_overlap": 0.5,
    "default_target_lang": "ces",
    # Audio preprocessing
    "preprocess_noise_gate": False,
    "preprocess_noise_gate_threshold": -40.0,  # dB
    "preprocess_normalize": True,
    "preprocess_normalize_target": -3.0,  # dB
    "preprocess_highpass": True,
    "preprocess_highpass_cutoff": 80,  # Hz
    "preprocess_auto_language": False,
    "num_beams": 1,
}

CONFIG_FILE = Path(__file__).parent.parent / "config.json"

SUPPORTED_LANGUAGES = {"ces", "eng", "rus", "ukr", "deu", "spa"}
FIXED_CONFIG_VALUES = {
    # The audio pipeline, Silero VAD and SeamlessM4T inference path are wired
    # around 16 kHz model audio. Device native sample rate is handled separately.
    "sample_rate": 16000,
}

runtime_config_lock = threading.RLock()

def load_config() -> dict:
    """Load config from JSON file, falling back to defaults."""
    config = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for key, value in saved.items():
                if key in DEFAULT_CONFIG:
                    config[key] = value
            logger.info(f"Config loaded from {CONFIG_FILE}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
    return config

def save_config(config_snapshot: dict | None = None):
    """Save current runtime_config to JSON file."""
    try:
        if config_snapshot is None:
            with runtime_config_lock:
                config_snapshot = dict(runtime_config)
        tmp_file = CONFIG_FILE.with_suffix(CONFIG_FILE.suffix + ".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(config_snapshot, f, indent=2, ensure_ascii=False)
        tmp_file.replace(CONFIG_FILE)
    except Exception as e:
        logger.error(f"Failed to save config: {e}")

# Validation ranges for config values
CONFIG_RANGES = {
    "silence_duration": (0.3, 3.0),
    "min_chunk_duration": (0.5, 5.0),
    "max_chunk_duration": (5.0, 30.0),
    "context_overlap": (0.0, 2.0),
    "sample_rate": (8000, 48000),
    "audio_channel": (0, 127),
    "preprocess_noise_gate_threshold": (-60.0, -10.0),
    "preprocess_normalize_target": (-20.0, 0.0),
    "preprocess_highpass_cutoff": (20, 300),
    "num_beams": (1, 5),
}


def _validate_loaded_config(config: dict) -> dict:
    """Validate config values against CONFIG_RANGES, reset out-of-range to defaults."""
    for key, fixed_value in FIXED_CONFIG_VALUES.items():
        if config.get(key) != fixed_value:
            logger.warning(f"Config '{key}' forced to {fixed_value}; runtime sample-rate changes are not supported")
            config[key] = fixed_value

    for key, (min_val, max_val) in CONFIG_RANGES.items():
        if key in config and config[key] is not None:
            try:
                val = config[key]
                if val < min_val or val > max_val:
                    logger.warning(f"Config '{key}' value {val} out of range ({min_val}-{max_val}), reset to default {DEFAULT_CONFIG[key]}")
                    config[key] = DEFAULT_CONFIG[key]
            except (TypeError, ValueError):
                logger.warning(f"Config '{key}' invalid type, reset to default {DEFAULT_CONFIG[key]}")
                config[key] = DEFAULT_CONFIG[key]

    # Validate types match defaults
    for key, value in config.items():
        if key in DEFAULT_CONFIG and value is not None and DEFAULT_CONFIG[key] is not None:
            expected_type = type(DEFAULT_CONFIG[key])
            if expected_type == float and isinstance(value, int) and not isinstance(value, bool):
                config[key] = float(value)
            elif not isinstance(value, expected_type):
                logger.warning(f"Config '{key}' wrong type ({type(value).__name__}), reset to default {DEFAULT_CONFIG[key]}")
                config[key] = DEFAULT_CONFIG[key]

    if config["min_chunk_duration"] > config["max_chunk_duration"]:
        logger.warning("Config min_chunk_duration > max_chunk_duration, resetting both to defaults")
        config["min_chunk_duration"] = DEFAULT_CONFIG["min_chunk_duration"]
        config["max_chunk_duration"] = DEFAULT_CONFIG["max_chunk_duration"]
    if config["context_overlap"] >= config["min_chunk_duration"]:
        logger.warning("Config context_overlap must be smaller than min_chunk_duration, resetting to default")
        config["context_overlap"] = DEFAULT_CONFIG["context_overlap"]
    return config


runtime_config = _validate_loaded_config(load_config())


def get_config_snapshot() -> dict:
    """Return a consistent copy of the current runtime configuration."""
    with runtime_config_lock:
        return dict(runtime_config)


def apply_runtime_config_updates(updates: dict) -> dict:
    """Atomically apply validated runtime config updates and return a snapshot."""
    with runtime_config_lock:
        runtime_config.update(updates)
        return dict(runtime_config)

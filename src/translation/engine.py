import time

import numpy as np
import torch
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor

from src.logging_handler import logger
from src.audio.preprocess import preprocess_audio

MODEL_NAME = "facebook/seamless-m4t-v2-large"


def load_model(device):
    """Load SeamlessM4T model. Returns (processor, model, dtype)."""
    logger.info(f"Loading model {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(MODEL_NAME)

    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # torch.compile on CUDA if available
    if device.type == "cuda":
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile()")
        except Exception as e:
            logger.warning(f"torch.compile() not available: {e}")

    logger.info("Model loaded.")
    return processor, model, dtype


def warmup_model(processor, model, device, dtype, sample_rate):
    """Run dummy inference to eliminate cold-start delay for main languages."""
    warmup_langs = ("ces", "eng", "rus", "ukr", "deu", "spa")
    logger.info(f"Warming up model ({', '.join(warmup_langs)})...")
    dummy_audio = np.zeros(sample_rate * 2, dtype=np.float32)
    inputs = processor(audio=dummy_audio, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=dtype) if v.dtype.is_floating_point else v.to(device=device)
              for k, v in inputs.items()}
    encoder_kwargs = {k: v for k, v in inputs.items() if k in ('input_features', 'attention_mask')}
    with torch.no_grad():
        encoder_out = model.get_encoder()(**encoder_kwargs)
        for lang in warmup_langs:
            model.generate(**inputs, encoder_outputs=encoder_out, tgt_lang=lang, num_beams=1, max_new_tokens=16)
    logger.info(f"Warm-up complete ({', '.join(warmup_langs)}).")


def translate_audio(audio_np, target_langs, processor, model, device, dtype, config, perf_metrics, perf_lock, translation_history, translation_history_lock):
    """Translate audio to multiple languages. Encoder runs once, decoder per language."""
    processed_audio = preprocess_audio(audio_np, config)

    sr = config["sample_rate"]
    inputs = processor(audio=processed_audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=dtype) if v.dtype.is_floating_point else v.to(device=device)
              for k, v in inputs.items()}

    # Run encoder once, reuse for all target languages
    encoder_kwargs = {k: v for k, v in inputs.items() if k in ('input_features', 'attention_mask')}
    results = {}
    with torch.no_grad():
        t_enc = time.time()
        encoder_out = model.get_encoder()(**encoder_kwargs)
        enc_ms = (time.time() - t_enc) * 1000
        logger.debug(f"Encoder: {enc_ms:.0f}ms")
        with perf_lock:
            perf_metrics["encoder_ms"].append(enc_ms)

        for lang in target_langs:
            try:
                t_dec = time.time()
                output_tokens = model.generate(
                    **inputs,
                    encoder_outputs=encoder_out,
                    tgt_lang=lang,
                    num_beams=1,
                    max_new_tokens=256,
                )
                dec_ms = (time.time() - t_dec) * 1000
                logger.debug(f"Decoder [{lang}]: {dec_ms:.0f}ms")
                with perf_lock:
                    perf_metrics["decoder_ms"].append(dec_ms)
                text = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True).strip()
                text = text.replace("#err", "")
                if text:
                    results[lang] = text
            except Exception as e:
                logger.error(f"Translation error for {lang}: {e}")

    # Record metrics and translation history
    with perf_lock:
        perf_metrics["total_translations"] += 1
        perf_metrics["last_inference_time"] = time.time()
    if results:
        with translation_history_lock:
            translation_history.append({
                "time": time.strftime("%H:%M:%S"),
                "translations": dict(results),
            })
    return results

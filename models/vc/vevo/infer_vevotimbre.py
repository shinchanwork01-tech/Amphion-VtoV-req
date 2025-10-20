# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from huggingface_hub import snapshot_download
import argparse  # <-- ADDED
import torch      # <-- ADDED (was missing from original file)

from models.vc.vevo.vevo_utils import *


def vevo_timbre(content_wav_path, reference_wav_path, output_path):
    gen_audio = inference_pipeline.inference_fm(
        src_wav_path=content_wav_path,
        timbre_ref_wav_path=reference_wav_path,
        flow_matching_steps=32,
    )
    save_audio(gen_audio, output_path=output_path)


if __name__ == "__main__":
    # ===== Argument Parser =====  <-- ADDED SECTION
    parser = argparse.ArgumentParser(description="VEVO Timbre Conversion using command-line arguments.")
    parser.add_argument('--source', required=True, type=str, help="Path to the source audio (content).")
    parser.add_argument('--reference', required=True, type=str, help="Path to the reference audio (timbre).")
    parser.add_argument('--output', required=True, type=str, help="Path to save the output audio.")
    args = parser.parse_args()
    
    # ===== Device =====
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ===== Content-Style Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["tokenizer/vq8192/*"],
    )
    tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

    # ===== Flow Matching Transformer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
    )

    fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

    # ===== Vocoder =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )

    vocoder_cfg_path = "./models/vc/vevo/config/Vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    # ===== Inference =====
    inference_pipeline = VevoInferencePipeline(
        content_style_tokenizer_ckpt_path=tokenizer_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )

    # Use arguments instead of hardcoded paths <-- MODIFIED SECTION
    vevo_timbre(args.source, args.reference, args.output)

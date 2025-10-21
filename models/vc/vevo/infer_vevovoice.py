# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# MODIFIED FOR EFFICIENT BATCH PROCESSING (FULL VOICE) - CORRECTED & VERIFIED

import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from huggingface_hub import snapshot_download
from models.vc.vevo.vevo_utils import *


# Core function using the correct method from the original file
def vevo_voice(inference_pipeline, content_wav_path, reference_wav_path, output_path):
    gen_audio = inference_pipeline.inference_ar_and_fm(
        src_wav_path=content_wav_path,
        src_text=None,
        style_ref_wav_path=reference_wav_path,
        timbre_ref_wav_path=reference_wav_path,
    )
    save_audio(gen_audio, output_path=output_path)


if __name__ == "__main__":
    # ===== 1. Argument Parser for Batch Processing =====
    parser = argparse.ArgumentParser(description="VEVO Full Voice Conversion Batch Processing")
    parser.add_argument('--csv', required=True, type=str, help="Path to the input CSV file.")
    parser.add_argument('--output_dir', default="Final_clips_voice", type=str, help="Directory to save the output files.")
    args = parser.parse_args()

    # ===== 2. LOAD ALL NECESSARY MODELS INTO VRAM (ONCE) =====
    # This section is now a direct copy of the original script's loading logic
    print("Loading all models for full voice conversion into VRAM...")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Content Tokenizer (Required for AR model)
    local_dir = snapshot_download(repo_id="amphion/Vevo", repo_type="model", cache_dir="./ckpts/Vevo", allow_patterns=["tokenizer/vq32/*"])
    content_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq32/hubert_large_l18_c32.pkl")

    # Content-Style Tokenizer
    local_dir = snapshot_download(repo_id="amphion/Vevo", repo_type="model", cache_dir="./ckpts/Vevo", allow_patterns=["tokenizer/vq8192/*"])
    content_style_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

    # Autoregressive Transformer (Required for AR model)
    local_dir = snapshot_download(repo_id="amphion/Vevo", repo_type="model", cache_dir="./ckpts/Vevo", allow_patterns=["contentstyle_modeling/Vq32ToVq8192/*"])
    ar_cfg_path = "./models/vc/vevo/config/Vq32ToVq8192.json"
    ar_ckpt_path = os.path.join(local_dir, "contentstyle_modeling/Vq32ToVq8192")

    # Flow Matching Transformer
    local_dir = snapshot_download(repo_id="amphion/Vevo", repo_type="model", cache_dir="./ckpts/Vevo", allow_patterns=["acoustic_modeling/Vq8192ToMels/*"])
    fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

    # Vocoder
    local_dir = snapshot_download(repo_id="amphion/Vevo", repo_type="model", cache_dir="./ckpts/Vevo", allow_patterns=["acoustic_modeling/Vocoder/*"])
    vocoder_cfg_path = "./models/vc/vevo/config/Vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    # Correct Inference Pipeline Initialization with ALL required arguments
    inference_pipeline = VevoInferencePipeline(
        content_tokenizer_ckpt_path=content_tokenizer_ckpt_path,
        content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
        ar_cfg_path=ar_cfg_path,
        ar_ckpt_path=ar_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )
    print("✅ Models loaded successfully.")

    # ===== 3. SETUP BATCH PROCESSING =====
    print(f"Reading CSV from '{args.csv}'...")
    df = pd.read_csv(args.csv)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved in '{args.output_dir}'")
    
    final_paths = []

    # ===== 4. LOOP AND PROCESS EFFICIENTLY =====
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing files"):
        source_path = row['clips']
        reference_path = row['path']
        output_filename = f"f_{index+1:04d}.wav"
        final_output_path = os.path.join(args.output_dir, output_filename)

        if not os.path.exists(source_path) or not os.path.exists(reference_path):
            final_paths.append(None)
            continue

        try:
            # Call the function for full voice conversion
            vevo_voice(inference_pipeline, source_path, reference_path, final_output_path)
            final_paths.append(final_output_path)
        except Exception as e:
            print(f"  -> ERROR processing row {index+1}: {e}")
            final_paths.append(None)

    # ===== 5. FINISH AND SAVE RESULTS =====
    df['final_voice'] = final_paths
    output_csv_path = os.path.join(args.output_dir, 'processed_data_voice.csv')
    df.to_csv(output_csv_path, index=False)

    print(f"\n✅ Batch processing complete.")
    print(f"Generated files are in the '{args.output_dir}' directory.")
    print(f"Updated CSV with final paths saved to: {output_csv_path}")

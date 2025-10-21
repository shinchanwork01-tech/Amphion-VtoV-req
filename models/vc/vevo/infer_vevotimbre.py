# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# MODIFIED FOR EFFICIENT BATCH PROCESSING

import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from huggingface_hub import snapshot_download
from models.vc.vevo.vevo_utils import *


# The core function remains the same, but we will pass the loaded pipeline to it
def vevo_timbre(inference_pipeline, content_wav_path, reference_wav_path, output_path):
    gen_audio = inference_pipeline.inference_fm(
        src_wav_path=content_wav_path,
        timbre_ref_wav_path=reference_wav_path,
        flow_matching_steps=32,
    )
    save_audio(gen_audio, output_path=output_path)


if __name__ == "__main__":
    # ===== 1. Argument Parser for Batch Processing =====
    parser = argparse.ArgumentParser(description="VEVO Timbre Conversion Batch Processing")
    parser.add_argument('--csv', required=True, type=str, help="Path to the input CSV file.")
    parser.add_argument('--output_dir', default="Final_clips", type=str, help="Directory to save the output files.")
    args = parser.parse_args()

    # ===== 2. LOAD MODELS INTO VRAM (ONCE) =====
    print("Loading models into VRAM... This will happen only once.")
    
    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Content-Style Tokenizer
    local_dir = snapshot_download(repo_id="amphion/Vevo", repo_type="model", cache_dir="./ckpts/Vevo", allow_patterns=["tokenizer/vq8192/*"])
    tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

    # Flow Matching Transformer
    local_dir = snapshot_download(repo_id="amphion/Vevo", repo_type="model", cache_dir="./ckpts/Vevo", allow_patterns=["acoustic_modeling/Vq8192ToMels/*"])
    fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

    # Vocoder
    local_dir = snapshot_download(repo_id="amphion/Vevo", repo_type="model", cache_dir="./ckpts/Vevo", allow_patterns=["acoustic_modeling/Vocoder/*"])
    vocoder_cfg_path = "./models/vc/vevo/config/Vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    # Create the single, persistent inference pipeline object
    inference_pipeline = VevoInferencePipeline(
        content_style_tokenizer_ckpt_path=tokenizer_ckpt_path,
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
            # Call the function with the already-loaded pipeline
            vevo_timbre(inference_pipeline, source_path, reference_path, final_output_path)
            final_paths.append(final_output_path)
        except Exception as e:
            print(f"  -> ERROR processing row {index+1}: {e}")
            final_paths.append(None)

    # ===== 5. FINISH AND SAVE RESULTS =====
    df['final'] = final_paths
    output_csv_path = os.path.join(args.output_dir, 'processed_data.csv')
    df.to_csv(output_csv_path, index=False)

    print(f"\n✅ Batch processing complete.")
    print(f"Generated files are in the '{args.output_dir}' directory.")
    print(f"Updated CSV with final paths saved to: {output_csv_path}")

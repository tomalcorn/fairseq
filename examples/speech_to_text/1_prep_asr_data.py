#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
from tqdm import tqdm

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text"]
SPLITS = ["train", "dev", "test"]

def load_transcriptions(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['audio', 'text'])
    # print(dict(zip(df['audio'], df['text'])))
    return dict(zip(df['audio'], df['text']))

def process(args):
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)

    # Load transcriptions
    transcriptions = load_transcriptions(args.transcription_file)

    # Extract features
    feature_root = out_root / "fbank80"
    feature_root.mkdir(exist_ok=True)

    audio_root = Path(args.audio_root)
    all_manifests = {}
    train_text = []

    for split in SPLITS:
        print(f"Processing {split} split...")
        split_dir = audio_root / split
        audio_files = list(split_dir.glob("*.wav"))  # Adjust file extension if needed

        print(f"Extracting log mel filter bank features for {split}...")
        manifest = {c: [] for c in MANIFEST_COLUMNS}

        for audio_file in tqdm(audio_files):
            sample_id = f"{split}-{audio_file.stem}"
            wav, sample_rate = torchaudio.load(audio_file)
            
            extract_fbank_features(
                wav, sample_rate, feature_root / f"{sample_id}.npy"
            )

            manifest["id"].append(sample_id)
            manifest["audio"].append(str(audio_file))
            manifest["n_frames"].append(wav.shape[1])
            print(f"Sample audio file name: {audio_file.name}")
            # print(f"Keys in transcriptions: {list(transcriptions.keys())[:5]}")  # Show first 5 keys
            print(f"Transcription for this file: {transcriptions.get(audio_file.name[:-4], 'NOT FOUND')}")
            manifest["tgt_text"].append(transcriptions.get(audio_file.name[:-4], "").lower())

        all_manifests[split] = manifest

        if split == "train":
            train_text.extend(manifest["tgt_text"])

    # Pack features into ZIP
    zip_path = out_root / "fbank80.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)

    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path)

    # Update manifests with correct audio paths and lengths, and save them
    for split, manifest in all_manifests.items():
        for i, sample_id in enumerate(manifest["id"]):
            manifest["audio"][i] = audio_paths[sample_id]
            manifest["n_frames"][i] = audio_lengths[sample_id]
        
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), out_root / f"{split}.tsv"
        )

    # Generate vocab
    vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm{args.vocab_type}{vocab_size}"
    
    # with NamedTemporaryFile(mode="w") as f:
    #     for t in train_text:
    #         f.write(t + "\n")
    #     gen_vocab(
    #     Path(f.name),
    #     out_root / spm_filename_prefix,
    #     args.vocab_type,
    #     args.vocab_size,
    #     )
    
    # Define the temporary file path
    temp_file = args.temp_file

    # Write the data to the file
    with open(temp_file, 'w') as f:
        for t in train_text:
            f.write(t + "\n")

    # Set read permissions for user, group, and others
    os.chmod(temp_file, 0o644)

    # Use the file in gen_vocab
    gen_vocab(
        Path(temp_file),
        out_root / spm_filename_prefix,
        args.vocab_type,
        args.vocab_size,
    )

    # Clean up the temporary file after you're done
    # os.remove(temp_file)

    # Generate config YAML
    gen_config_yaml(
        out_root,
        spm_filename=spm_filename_prefix + ".model",
        specaugment_policy="ld"
    )

    # Clean up
    shutil.rmtree(feature_root)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument("--audio-root", required=True, type=str, help="Root directory containing train, dev, and test folders")
    parser.add_argument("--transcription-file", required=True, type=str, help="TSV file containing transcriptions")
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    )
    parser.add_argument("--vocab-size", default=10000, type=int)
    parser.add_argument("--temp-file", default="/work/tc062/tc062/s2517781/tmp.txt", type=str)
    args = parser.parse_args()

    process(args)

if __name__ == "__main__":
    main()
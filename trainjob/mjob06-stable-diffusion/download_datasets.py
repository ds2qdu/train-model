#!/usr/bin/env python3
"""
Stable Diffusion Training Dataset Downloader
Downloads sample datasets from Hugging Face to local/NFS storage

Usage:
    python download_datasets.py --output_dir "Y:/mlteam-stable-diffusion-storage-pvc-.../data/train"
    python download_datasets.py --output_dir "./data/train" --max_images 100
    python download_datasets.py --dataset pokemon --max_images 50
"""

import argparse
from pathlib import Path


def download_single_dataset(data_path, dataset_name, max_images):
    """Download a single dataset from Hugging Face"""
    from PIL import Image
    from datasets import load_dataset

    count = 0

    if dataset_name == "pokemon":
        # Pokemon dataset - good for LoRA training
        ds = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
        print(f"Pokemon dataset loaded: {len(ds)} images available")

        for i, item in enumerate(ds):
            if i >= max_images:
                break

            img = item['image']
            caption = item['text']

            # Save image
            img_path = data_path / f"pokemon_{i:04d}.png"
            img.save(img_path)

            # Save caption
            caption_path = data_path / f"pokemon_{i:04d}.txt"
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            count += 1

            if (i + 1) % 10 == 0:
                print(f"  Pokemon: {i + 1}/{min(max_images, len(ds))} images...")

    elif dataset_name == "cat":
        # Cat toy dataset - good for DreamBooth
        ds = load_dataset("diffusers/cat_toy_example", split="train")
        print(f"Cat toy dataset loaded: {len(ds)} images available")

        for i, item in enumerate(ds):
            if i >= max_images:
                break

            img = item['image']

            # Save image
            img_path = data_path / f"cat_{i:04d}.png"
            img.save(img_path)

            # Create caption
            caption = "a photo of sks cat toy"
            caption_path = data_path / f"cat_{i:04d}.txt"
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            count += 1

    elif dataset_name == "anime":
        # Anime/cartoon dataset
        ds = load_dataset("Norod78/cartoon-blip-captions", split="train")
        print(f"Cartoon/Anime dataset loaded: {len(ds)} images available")

        for i, item in enumerate(ds):
            if i >= max_images:
                break

            img = item['image']
            caption = item.get('text', 'an anime style illustration')

            img_path = data_path / f"anime_{i:04d}.png"
            img.save(img_path)

            caption_path = data_path / f"anime_{i:04d}.txt"
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            count += 1

            if (i + 1) % 10 == 0:
                print(f"  Anime: {i + 1}/{min(max_images, len(ds))} images...")

    elif dataset_name == "art":
        # Art/painting dataset - use streaming to avoid downloading 35GB+
        from datasets import load_dataset_builder

        # Get label mappings from dataset info (without downloading data)
        print("  Loading WikiArt label mappings...")
        builder = load_dataset_builder("huggan/wikiart")
        features = builder.info.features

        artist_names = features['artist'].names if hasattr(features['artist'], 'names') else None
        style_names = features['style'].names if hasattr(features['style'], 'names') else None
        genre_names = features['genre'].names if hasattr(features['genre'], 'names') else None

        print(f"  Artists: {len(artist_names) if artist_names else 'N/A'}")
        print(f"  Styles: {len(style_names) if style_names else 'N/A'}")
        print(f"  Genres: {len(genre_names) if genre_names else 'N/A'}")

        # Now stream the actual data
        ds = load_dataset("huggan/wikiart", split="train", streaming=True)
        print(f"WikiArt dataset (streaming mode)...")

        idx = 0
        for item in ds:
            if idx >= max_images:
                break
            try:
                img = item['image']

                # Get actual text labels from indices
                artist_idx = item.get('artist', 0)
                style_idx = item.get('style', 0)
                genre_idx = item.get('genre', 0)

                artist = artist_names[artist_idx] if artist_names and artist_idx < len(artist_names) else "unknown artist"
                style = style_names[style_idx] if style_names and style_idx < len(style_names) else "painting"
                genre = genre_names[genre_idx] if genre_names and genre_idx < len(genre_names) else "artwork"

                # Clean up names (replace underscores)
                artist = artist.replace('_', ' ')
                style = style.replace('_', ' ')
                genre = genre.replace('_', ' ')

                img_path = data_path / f"art_{idx:04d}.png"
                img.save(img_path)

                # Create descriptive caption
                caption = f"a {style} {genre} painting by {artist}"
                caption_path = data_path / f"art_{idx:04d}.txt"
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)

                idx += 1
                count += 1
                if idx % 50 == 0:
                    print(f"  Art: {idx}/{max_images} images...")
            except Exception as e:
                print(f"  Skipping art image: {e}")
                continue

    return count


def main():
    parser = argparse.ArgumentParser(description='Download training datasets for Stable Diffusion')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for downloaded images')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['pokemon', 'cat', 'anime', 'art', 'all'],
                        help='Dataset to download (default: all)')
    parser.add_argument('--max_images', type=int, default=100,
                        help='Maximum images per dataset (default: 100)')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Stable Diffusion Training Dataset Downloader")
    print("=" * 60)
    print(f"  Output: {output_path}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Max images per dataset: {args.max_images}")
    print("=" * 60)
    print()

    # Install datasets if not available
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing 'datasets' library...")
        import subprocess
        subprocess.run(['pip', 'install', 'datasets', 'Pillow'], check=True)
        from datasets import load_dataset

    # Download datasets
    if args.dataset == "all":
        all_datasets = ["pokemon", "cat", "anime", "art"]
        total_count = 0

        for ds_name in all_datasets:
            print(f"\n>>> Downloading {ds_name.upper()} dataset...")
            try:
                count = download_single_dataset(output_path, ds_name, args.max_images)
                total_count += count
                print(f"    [OK] {ds_name}: {count} images downloaded")
            except Exception as e:
                print(f"    [FAIL] {ds_name}: {e}")

        print()
        print("=" * 60)
        print(f"  Download Complete!")
        print(f"  Total: {total_count} images")
        print(f"  Location: {output_path}")
        print("=" * 60)
    else:
        print(f"\n>>> Downloading {args.dataset.upper()} dataset...")
        count = download_single_dataset(output_path, args.dataset, args.max_images)
        print()
        print("=" * 60)
        print(f"  Download Complete!")
        print(f"  {args.dataset}: {count} images")
        print(f"  Location: {output_path}")
        print("=" * 60)

    # Show sample files
    print("\nSample files:")
    files = list(output_path.glob("*.png"))[:5]
    for f in files:
        txt_file = f.with_suffix('.txt')
        caption = ""
        if txt_file.exists():
            with open(txt_file, 'r', encoding='utf-8') as tf:
                caption = tf.read()[:50] + "..." if len(tf.read()) > 50 else tf.read()
                tf.seek(0)
                caption = tf.read()[:50]
        print(f"  - {f.name}: {caption}...")


if __name__ == '__main__':
    main()

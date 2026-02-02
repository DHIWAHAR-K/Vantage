"""
Download text-to-SQL datasets via direct URLs (Spider 1.0, Spider2, BIRD dev/train, WikiSQL).
"""

import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# URLs that work for direct download
SPIDER2_URL = "https://github.com/xlang-ai/Spider2/archive/refs/heads/main.zip"
BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
BIRD_TRAIN_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
WIKISQL_URL = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"
# Spider 1.0: Google Drive (use gdown); id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J
SPIDER1_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"


def download_file(url: str, dest: Path, desc: str = "Downloading") -> Path:
    """Download a file from URL to dest. Returns path to downloaded file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"{desc}...")
    try:
        urlretrieve(url, dest)
        print(f"  ✓ Saved to {dest}")
        return dest
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def download_spider1(output_dir: Path) -> None:
    """Download Spider 1.0 from Google Drive (requires gdown)."""
    base = output_dir / "text2sql" if (output_dir / "text2sql").is_dir() else output_dir
    base.mkdir(parents=True, exist_ok=True)
    zip_path = base / "spider1.zip"
    if zip_path.exists():
        print("  spider1.zip already exists, skipping download")
        return
    print("Downloading Spider 1.0 (Google Drive)...")
    try:
        subprocess.run(
            [sys.executable, "-m", "gdown", f"--id={SPIDER1_GDRIVE_ID}", "-O", str(zip_path)],
            check=True,
        )
        print(f"  ✓ Saved to {zip_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  ✗ Failed: {e}")
        print("  Install with: pip install gdown")
        print("  Or download manually: https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing")
        raise


def download_spider2(output_dir: Path) -> None:
    """Download Spider2 dataset (GitHub archive)."""
    base = output_dir / "text2sql" if (output_dir / "text2sql").is_dir() else output_dir
    base.mkdir(parents=True, exist_ok=True)
    zip_path = base / "spider2.zip"
    download_file(SPIDER2_URL, zip_path, desc="Downloading Spider2")
    print("  Extracting...")
    spider2_dir = base / "spider2"
    spider2_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(spider2_dir)
    zip_path.unlink()
    print(f"✓ Spider2 ready at {spider2_dir}")


def download_bird_dev(output_dir: Path) -> None:
    """Download BIRD dev set (Aliyun OSS)."""
    base = output_dir / "text2sql" if (output_dir / "text2sql").is_dir() else output_dir
    base.mkdir(parents=True, exist_ok=True)
    zip_path = base / "bird_dev.zip"
    download_file(BIRD_DEV_URL, zip_path, desc="Downloading BIRD dev")
    print("  Extracting...")
    bird_dir = base / "bird"
    bird_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(bird_dir)
    zip_path.unlink()
    print(f"✓ BIRD dev ready at {bird_dir}")


def download_bird_train(output_dir: Path) -> None:
    """Download BIRD train set (Aliyun OSS)."""
    base = output_dir / "text2sql" if (output_dir / "text2sql").is_dir() else output_dir
    base.mkdir(parents=True, exist_ok=True)
    zip_path = base / "bird_train.zip"
    download_file(BIRD_TRAIN_URL, zip_path, desc="Downloading BIRD train")
    print("  Extracting...")
    bird_dir = base / "bird"
    bird_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(bird_dir)
    zip_path.unlink()
    print(f"✓ BIRD train ready at {bird_dir}")


def download_wikisql(output_dir: Path) -> None:
    """Download WikiSQL data (Salesforce GitHub)."""
    base = output_dir / "text2sql" if (output_dir / "text2sql").is_dir() else output_dir
    base.mkdir(parents=True, exist_ok=True)
    wikisql_dir = base / "wikisql"
    wikisql_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = base / "data.tar.bz2"
    download_file(WIKISQL_URL, tarball_path, desc="Downloading WikiSQL")
    print("  Extracting...")
    with tarfile.open(tarball_path, "r:bz2") as tf:
        tf.extractall(wikisql_dir)
    tarball_path.unlink()
    print(f"✓ WikiSQL ready at {wikisql_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download text-to-SQL datasets (Spider 1.0, Spider2, BIRD dev/train, WikiSQL)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["spider1", "spider2", "bird_dev", "bird_train", "wikisql"],
        choices=["spider1", "spider2", "bird_dev", "bird_train", "bird", "wikisql", "all"],
        help="Datasets to download",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for datasets",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets
    if "all" in datasets:
        datasets = ["spider1", "spider2", "bird_dev", "bird_train", "wikisql"]
    if "bird" in datasets:
        datasets = [d for d in datasets if d != "bird"] + ["bird_dev", "bird_train"]

    print(f"Downloading to {output_dir}")
    print("=" * 50)

    for name in datasets:
        if name == "spider1":
            download_spider1(output_dir)
        elif name == "spider2":
            download_spider2(output_dir)
        elif name == "bird_dev":
            download_bird_dev(output_dir)
        elif name == "bird_train":
            download_bird_train(output_dir)
        elif name == "wikisql":
            download_wikisql(output_dir)
        else:
            print(f"Unknown dataset: {name}")
        print()

    print("=" * 50)
    print("Dataset download complete!")
    print(f"Datasets saved to: {output_dir}")


if __name__ == "__main__":
    main()

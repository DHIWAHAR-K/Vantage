"""
Download text-to-SQL datasets via direct URLs (Spider2, BIRD, WikiSQL).
"""

import argparse
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# URLs that work for direct download
SPIDER2_URL = "https://github.com/xlang-ai/Spider2/archive/refs/heads/main.zip"
BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
WIKISQL_URL = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"


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


def download_spider2(output_dir: Path) -> None:
    """Download Spider2 dataset (GitHub archive)."""
    spider2_dir = output_dir / "spider2"
    spider2_dir.mkdir(parents=True, exist_ok=True)
    zip_path = spider2_dir / "spider2.zip"

    download_file(SPIDER2_URL, zip_path, desc="Downloading Spider2")

    print("  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(spider2_dir)
    zip_path.unlink()
    print(f"✓ Spider2 ready at {spider2_dir}")


def download_bird(output_dir: Path) -> None:
    """Download BIRD dev set (Aliyun OSS)."""
    bird_dir = output_dir / "bird"
    bird_dir.mkdir(parents=True, exist_ok=True)
    zip_path = bird_dir / "bird_dev.zip"

    download_file(BIRD_DEV_URL, zip_path, desc="Downloading BIRD dev")

    print("  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(bird_dir)
    zip_path.unlink()
    print(f"✓ BIRD dev ready at {bird_dir}")


def download_wikisql(output_dir: Path) -> None:
    """Download WikiSQL data (Salesforce GitHub)."""
    wikisql_dir = output_dir / "wikisql"
    wikisql_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = wikisql_dir / "data.tar.bz2"

    download_file(WIKISQL_URL, tarball_path, desc="Downloading WikiSQL")

    print("  Extracting...")
    with tarfile.open(tarball_path, "r:bz2") as tf:
        tf.extractall(wikisql_dir)
    tarball_path.unlink()
    print(f"✓ WikiSQL ready at {wikisql_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download text-to-SQL datasets (Spider2, BIRD, WikiSQL)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["spider2", "bird", "wikisql"],
        choices=["spider2", "bird", "wikisql", "all"],
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
        datasets = ["spider2", "bird", "wikisql"]

    print(f"Downloading to {output_dir}")
    print("=" * 50)

    for name in datasets:
        if name == "spider2":
            download_spider2(output_dir)
        elif name == "bird":
            download_bird(output_dir)
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

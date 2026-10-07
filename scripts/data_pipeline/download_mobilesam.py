#!/usr/bin/env python
# Download MobileSAM weights directly (faster than git clone)

import os
import urllib.request

# Try different mirrors
MIRRORS = [
    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
    "https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt",
    "https://cloud.tsinghua.edu.cn/f/a390a9652a2743e0a5f6/?dl=1",
]

def download_with_progress(url, destination):
    """Download file with progress bar"""
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = downloaded * 100 / total_size
        if total_size > 0:
            print("\r  Downloading... %.1f%%" % percent, end='')

    try:
        print("Trying: %s" % url)
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print("\n  Downloaded successfully!")
        return True
    except Exception as e:
        print("\n  Failed: %s" % e)
        return False

def main():
    # Download to current directory
    output_path = "mobile_sam.pt"

    if os.path.exists(output_path):
        print("MobileSAM weights already exist: %s" % output_path)
        return

    print("Downloading MobileSAM weights...")
    for mirror in MIRRORS:
        if download_with_progress(mirror, output_path):
            print("Saved to: %s" % os.path.abspath(output_path))
            return

    print("All mirrors failed. Please download manually.")
    print("You can download from: https://github.com/ChaoningZhang/MobileSAM")

if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

import requests


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python test_decode.py <image1> <image2>")
        sys.exit(1)

    image_paths = [Path(sys.argv[1]), Path(sys.argv[2])]
    for path in image_paths:
        if not path.exists() or not path.is_file():
            print(f"File not found: {path}")
            sys.exit(1)

    files = []
    handles = []
    try:
        for path in image_paths:
            fh = path.open("rb")
            handles.append(fh)
            files.append(("frames", (path.name, fh, "image/jpeg")))

        response = requests.post("http://127.0.0.1:8000/decode", files=files, timeout=30)
        response.raise_for_status()
        print(response.json())
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        sys.exit(1)
    finally:
        for fh in handles:
            fh.close()


if __name__ == "__main__":
    main()

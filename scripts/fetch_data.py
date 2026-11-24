"""Retrieve pinned public inputs and the modWorm checkout."""

import argparse
from hashlib import file_digest
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
STATE_SPACE = "https://api.bitbucket.org/2.0/repositories/tosifahamed/behavioral-state-space/src/ad125e08db4537379371d89e490bf913f7b5e510/data"
BAAI = "https://raw.githubusercontent.com/Jessie940611/BAAIWorm/b99ca5ab9c00aef739fceb1cf007bc50b06f2ccc/eworm/ghost_in_mesh_sim/data/tuned/video_offline"
MOD_REV = "10912b4e09c72ba3430573f0b93a48c6e86b5614"
ASSETS = {
    "crawl.mat": (
        f"{STATE_SPACE}/crawl.mat",
        "184ac7bd894c090c290bd286a32e0ba2fb284daa54a1f36697c3f981e5b34c1c",
    ),
    "EigenWorms.mat": (
        f"{STATE_SPACE}/EigenWorms.mat",
        "29fb788cad5308b80c414154c95f76469d32127562784e2bfcd05df6b902c495",
    ),
    "NeuronConnect.xls": (
        "https://www.wormatlas.org/images/NeuronConnect.xls",
        "120c2c6332050a2d1494c19c687f447ed65620ad0db5f8b732189aa10e5162f1",
    ),
    "NeuronFixedPoints.xls": (
        "https://www.wormatlas.org/images/NeuronFixedPoints.xls",
        "7769758242d6ecfb41da0860c50dea1acf042a060d12fbb1ac2f1c4888ea9703",
    ),
    "cook2019_corrected2020.xlsx": (
        "https://wormwiring.org/si/SI%205%20Connectome%20adjacency%20matrices,%20corrected%20July%202020.xlsx",
        "1f4fdbf84746b69b49a8da0816f52787860ce349b638dce37924ba80f90c70c9",
    ),
    "baaiworm/gt_eworm.muscle-220428-152601.npy": (
        f"{BAAI}/gt_eworm.muscle-220428-152601.npy",
        "faf026dfb5e1863451f1b6124c1c49d83363991c3bf07a42213a78b6a58cad28",
    ),
    "baaiworm/video_offline_eworm.muscle-220428-152601.npy": (
        f"{BAAI}/video_offline_eworm.muscle-220428-152601.npy",
        "817e800c024c218ff9cb7f2a69ab81c9028124abb5dbb5ce261e9775c7484e69",
    ),
    "baaiworm/video_offline_config.json": (
        f"{BAAI}/video_offline_config.json",
        "7fbcb0a95d72823fa9b31aa1919f1896b6369714c35ca69f175415a66abf8ce2",
    ),
}


def sha(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def fetch(destination, url, checksum):
    if destination.exists() and sha(destination) == checksum:
        print(f"Verified {destination.relative_to(RAW)}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as file:
        pending = Path(file.name)
    try:
        command = ["curl", "-fsSL", "--max-time", "90", url, "-o", str(pending)]
        result = subprocess.run(command)
        if result.returncode == 60 and url.startswith("https://www.wormatlas.org/"):
            # Work around the legacy server's certificate chain. The pinned hash must match.
            subprocess.run([*command, "--insecure"], check=True)
        elif result.returncode:
            result.check_returncode()
        if sha(pending) != checksum:
            raise ValueError(f"SHA-256 mismatch for {destination.name}")
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)
    print(f"Downloaded and verified {destination.relative_to(RAW)}")


def checkout_modworm():
    path = ROOT / "external" / "modWorm"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                           text=True).strip()
        if revision != MOD_REV:
            raise ValueError(f"Existing modWorm checkout has revision {revision}. Expected {MOD_REV}.")
        print(f"Verified modWorm {revision}")
        return
    with tempfile.TemporaryDirectory(dir=path.parent) as stage:
        pending = Path(stage) / "modWorm"
        subprocess.run(["git", "init", str(pending)], check=True)
        subprocess.run(["git", "-C", str(pending), "fetch", "--depth", "1",
                        "https://github.com/shlizee/modWorm.git", MOD_REV], check=True)
        subprocess.run(["git", "-C", str(pending), "checkout", "--detach", "FETCH_HEAD"], check=True)
        pending.rename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-only", action="store_true")
    args = parser.parse_args()
    selected = ("crawl.mat", "EigenWorms.mat") if args.behavior_only else ASSETS
    for name in selected:
        fetch(RAW / name, *ASSETS[name])
    if not args.behavior_only:
        checkout_modworm()


if __name__ == "__main__":
    main()

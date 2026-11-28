"""Restore either shadow benchmark package using an existing Namaqualand object store."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import zipfile


def references(value):
    if isinstance(value, dict):
        if "hash" in value and "size" in value:
            yield value["hash"], value["size"]
        else:
            for item in value.values():
                yield from references(item)
    elif isinstance(value, list):
        for item in value:
            yield from references(item)


def restore_historical_mesh(candidates, size, digest, target):
    # These two benchmark manifests pin meshes cooked before the glTF tangent.w fix.
    # Recover only the exact archived bytes. Never modify the current object store.
    for path in candidates.get(size, ()):
        with path.open("rb") as src:
            header = src.read(16)
            if len(header) != 16 or header[:8] != b"RMESH001":
                continue
            vertices, indices = struct.unpack_from("<II", header, 8)
            if size != 16 + vertices * 48 + indices * 4:
                continue
            data = bytearray(header + src.read())
        # Little-endian float sign bit, fourth tangent component at vertex offset 36.
        signs = slice(16 + 39, 16 + vertices * 48, 48)
        data[signs] = data[signs].translate(bytes(i ^ 128 for i in range(256)))
        if hashlib.sha256(data).hexdigest() == digest:
            target.write_bytes(data)
            return
    raise FileNotFoundError(f"No object matching historical mesh {digest}")


def restore(base, destination, archive, mode):
    with zipfile.ZipFile(archive) as evidence:
        manifest = evidence.read(f"{mode}/scene.json")
        destination.mkdir(parents=True, exist_ok=False)
        objects = destination / "objects"
        objects.mkdir()
        members = set(evidence.namelist())
        candidates = None
        for digest, size in sorted(set(references(json.loads(manifest)))):
            if len(digest) != 64 or bytes.fromhex(digest).hex() != digest:
                raise ValueError(f"Invalid object digest {digest!r}")
            target = objects / digest
            member = f"objects/{digest}"
            if member in members:
                with evidence.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
            else:
                source = base / "objects" / digest
                if source.is_file():
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copyfile(source, target)
                else:
                    if candidates is None:
                        candidates = {}
                        for obj in (base / "objects").iterdir():
                            candidates.setdefault(obj.stat().st_size, []).append(obj)
                    restore_historical_mesh(candidates, size, digest, target)
            with target.open("rb") as blob:
                if target.stat().st_size != size or hashlib.file_digest(blob, "sha256").hexdigest() != digest:
                    raise ValueError(f"Object mismatch: {target}")
        (destination / "scene.json").write_bytes(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="existing cooked package")
    parser.add_argument("destination", type=Path, help="new output directory")
    parser.add_argument("--mode", choices=["walk", "scene"], default="walk")
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parents[2]
                        / "assets/verification/namaqualand-near-csm-evidence.zip")
    args = parser.parse_args()
    restore(args.base, args.destination, args.archive, args.mode)

"""Create a compact RadioML 2018.01A subset without downloading the full HDF5."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import BinaryIO

import h5py
import numpy as np
import requests


SOURCE_URL = (
    "https://huggingface.co/datasets/Katherinezml/RML2018.01A/resolve/main/"
    "2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
)
LICENSE_URL = (
    "https://huggingface.co/datasets/Katherinezml/RML2018.01A/resolve/main/"
    "2018.01.OSC.0001_1024x2M.h5/2018.01/LICENSE.TXT"
)
SOURCE_SIZE = 21_449_148_312
SOURCE_SHA256 = "e3dd0bef66a3426959ee66a1709a8c0a95d4f8395d18aaf6f1214bdbc763bd38"
CLASS_NAMES = [
    "32PSK", "16APSK", "32QAM", "FM", "GMSK", "32APSK", "OQPSK", "8ASK",
    "BPSK", "8PSK", "AM-SSB-SC", "4ASK", "16PSK", "64APSK", "128QAM",
    "128APSK", "AM-DSB-SC", "AM-SSB-WC", "64QAM", "QPSK", "256QAM",
    "AM-DSB-WC", "OOK", "16QAM",
]
ALL_SNRS = tuple(range(-20, 32, 2))
FRAMES_PER_CONDITION = 4096
FRAME_SHAPE = (1024, 2)


class HTTPRangeReader(io.RawIOBase):
    """Seekable HTTP reader that refuses full-file responses and enforces a byte budget."""

    def __init__(self, url: str, max_transfer_bytes: int) -> None:
        super().__init__()
        self.session = requests.Session()
        head = self.session.head(url, allow_redirects=True, timeout=(15, 60))
        head.raise_for_status()
        self.url = head.url
        self.size = int(head.headers["Content-Length"])
        if self.size != SOURCE_SIZE:
            raise OSError(f"Remote size changed: expected {SOURCE_SIZE}, received {self.size}.")
        linked_headers = head.history[0].headers if head.history else head.headers
        linked_hash = linked_headers.get("X-Linked-ETag", "").strip('"')
        if linked_hash and linked_hash != SOURCE_SHA256:
            raise OSError(f"Remote source hash changed: {linked_hash}")
        self.position = 0
        self.transferred = 0
        self.max_transfer_bytes = max_transfer_bytes

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self.position + offset
        elif whence == os.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0 or position > self.size:
            raise OSError(f"Seek outside remote file: {position}")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            raise OSError("Unbounded reads are disabled to protect local storage and bandwidth.")
        if size == 0 or self.position >= self.size:
            return b""
        requested = min(size, self.size - self.position)
        if self.transferred + requested > self.max_transfer_bytes:
            raise OSError(
                f"Transfer ceiling exceeded ({self.max_transfer_bytes / 1_048_576:.0f} MiB)."
            )
        start = self.position
        stop = start + requested - 1
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{stop}"},
            allow_redirects=False,
            stream=True,
            timeout=(15, 300),
        )
        try:
            if response.status_code != 206:
                raise OSError(
                    f"Server did not honor byte range {start}-{stop}; HTTP {response.status_code}."
                )
            content = response.content
        finally:
            response.close()
        if len(content) != requested:
            raise OSError(f"Short range response: expected {requested}, received {len(content)} bytes.")
        self.position += requested
        self.transferred += requested
        return content

    def readinto(self, buffer: BinaryIO) -> int:
        content = self.read(len(buffer))
        buffer[: len(content)] = content
        return len(content)

    def close(self) -> None:
        if not self.closed:
            self.session.close()
        super().close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _condition_ranges(
    selected_snrs: list[int], samples_per_condition: int, seed: int
) -> list[tuple[int, int, int, int]]:
    if not 1 <= samples_per_condition <= FRAMES_PER_CONDITION:
        raise ValueError(f"samples-per-condition must be in [1, {FRAMES_PER_CONDITION}].")
    if not selected_snrs or set(selected_snrs) - set(ALL_SNRS):
        raise ValueError(f"SNR values must be a non-empty subset of {ALL_SNRS}.")
    if len(selected_snrs) != len(set(selected_snrs)):
        raise ValueError("SNR values must not contain duplicates.")
    rng = np.random.default_rng(seed)
    ranges = []
    for class_id in range(len(CLASS_NAMES)):
        for snr_db in sorted(selected_snrs):
            snr_position = ALL_SNRS.index(snr_db)
            condition_start = (
                class_id * len(ALL_SNRS) + snr_position
            ) * FRAMES_PER_CONDITION
            offset = int(rng.integers(FRAMES_PER_CONDITION - samples_per_condition + 1))
            start = condition_start + offset
            ranges.append((class_id, snr_db, start, start + samples_per_condition))
    return ranges


def _validate_source(handle: h5py.File) -> None:
    missing = {"X", "Y", "Z"} - set(handle.keys())
    if missing:
        raise ValueError(f"Remote source is missing datasets: {sorted(missing)}")
    expected_count = len(CLASS_NAMES) * len(ALL_SNRS) * FRAMES_PER_CONDITION
    if handle["X"].shape != (expected_count, *FRAME_SHAPE):
        raise ValueError(f"Unexpected X shape: {handle['X'].shape}")
    if handle["Y"].shape != (expected_count, len(CLASS_NAMES)):
        raise ValueError(f"Unexpected Y shape: {handle['Y'].shape}")
    if handle["Z"].shape not in ((expected_count,), (expected_count, 1)):
        raise ValueError(f"Unexpected Z shape: {handle['Z'].shape}")


def _inspect_source(max_transfer_mib: int) -> None:
    reader = HTTPRangeReader(SOURCE_URL, max_transfer_mib * 1024 * 1024)
    try:
        with h5py.File(reader, "r") as source:
            _validate_source(source)
            for name in ("X", "Y", "Z"):
                dataset = source[name]
                print(
                    f"{name}: shape={dataset.shape}, dtype={dataset.dtype}, chunks={dataset.chunks}, "
                    f"compression={dataset.compression}, offset={dataset.id.get_offset()}",
                    flush=True,
                )
        print(f"Remote bytes transferred: {reader.transferred}", flush=True)
    finally:
        reader.close()


def _create_subset(
    output: Path,
    selected_snrs: list[int],
    samples_per_condition: int,
    seed: int,
    max_transfer_mib: int,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    ranges = _condition_ranges(selected_snrs, samples_per_condition, seed)
    selected_indices = np.concatenate(
        [np.arange(start, stop, dtype=np.int64) for _, _, start, stop in ranges]
    )
    expected_labels = np.repeat(
        [class_id for class_id, _, _, _ in ranges], samples_per_condition
    )
    expected_snrs = np.repeat(
        [snr_db for _, snr_db, _, _ in ranges], samples_per_condition
    )
    total = len(selected_indices)

    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=output.stem + ".", suffix=".partial.hdf5", dir=output.parent, delete=False
    )
    temporary = Path(temporary_handle.name)
    temporary_handle.close()
    reader: HTTPRangeReader | None = None
    try:
        print("Opening the full-file mirror in byte-range mode (no local source cache).", flush=True)
        reader = HTTPRangeReader(SOURCE_URL, max_transfer_mib * 1024 * 1024)
        with h5py.File(reader, "r") as source:
            _validate_source(source)
            print(
                f"Verified remote schema: X={source['X'].shape}, Y={source['Y'].shape}, "
                f"Z={source['Z'].shape}",
                flush=True,
            )
            print("Reading only selected label/SNR ranges for validation...", flush=True)
            label_parts = []
            snr_parts = []
            for condition_number, (class_id, snr_db, start, stop) in enumerate(ranges, 1):
                labels = np.asarray(source["Y"][start:stop])
                snrs = np.asarray(source["Z"][start:stop]).reshape(-1)
                if not np.all(np.argmax(labels, axis=1) == class_id):
                    raise ValueError(
                        f"Source label ordering failed for class {class_id} at {start}:{stop}."
                    )
                if not np.all(snrs == snr_db):
                    raise ValueError(
                        f"Source SNR ordering failed for {snr_db} dB at {start}:{stop}."
                    )
                label_parts.append(labels)
                snr_parts.append(snrs)
                if condition_number % len(selected_snrs) == 0:
                    print(
                        f"Validated metadata for class {class_id + 1:02d}/{len(CLASS_NAMES)}",
                        flush=True,
                    )
            selected_labels = np.concatenate(label_parts)
            selected_snr_values = np.concatenate(snr_parts)
            if not np.all(np.argmax(selected_labels, axis=1) == expected_labels):
                raise ValueError("Selected modulation labels do not match documented source ordering.")
            if not np.all(selected_snr_values == expected_snrs):
                raise ValueError("Selected SNR labels do not match documented source ordering.")

            with h5py.File(temporary, "w") as target:
                x_output = target.create_dataset(
                    "X",
                    shape=(total, *FRAME_SHAPE),
                    dtype=source["X"].dtype,
                    chunks=(min(samples_per_condition, 64), *FRAME_SHAPE),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                target.create_dataset("Y", data=selected_labels)
                target.create_dataset("Z", data=selected_snr_values.reshape(-1, 1))
                target.create_dataset("source_index", data=selected_indices)
                cursor = 0
                for condition_number, (class_id, _, start, stop) in enumerate(ranges, 1):
                    count = stop - start
                    x_output[cursor : cursor + count] = source["X"][start:stop]
                    cursor += count
                    if condition_number % len(selected_snrs) == 0:
                        print(
                            f"Copied class {class_id + 1:02d}/{len(CLASS_NAMES)} "
                            f"({cursor}/{total} I/Q windows)",
                            flush=True,
                        )
                target.attrs["source_url"] = SOURCE_URL
                target.attrs["source_full_file_sha256"] = SOURCE_SHA256
                target.attrs["class_names"] = json.dumps(CLASS_NAMES)
                target.attrs["selected_snrs_db"] = json.dumps(sorted(selected_snrs))
                target.attrs["samples_per_modulation_snr"] = samples_per_condition
                target.attrs["selection_seed"] = seed
                target.attrs["license"] = "CC BY-NC-SA 4.0"
                target.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        transferred = reader.transferred
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if reader is not None:
            reader.close()

    return {
        "source_url": SOURCE_URL,
        "source_full_file_size_bytes": SOURCE_SIZE,
        "source_full_file_sha256": SOURCE_SHA256,
        "remote_bytes_transferred": transferred,
        "output_file": output.name,
        "output_size_bytes": output.stat().st_size,
        "output_sha256": _sha256(output),
        "X_shape": [total, *FRAME_SHAPE],
        "Y_shape": [total, len(CLASS_NAMES)],
        "Z_shape": [total, 1],
        "class_names": CLASS_NAMES,
        "selected_snrs_db": sorted(selected_snrs),
        "samples_per_modulation_snr": samples_per_condition,
        "selection_seed": seed,
        "selection_method": "one deterministic contiguous block per modulation/SNR condition",
        "license": "CC BY-NC-SA 4.0",
        "license_url": "https://www.deepsig.ai/datasets/",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snrs", nargs="+", type=int, default=[-20, -10, 0, 10, 20, 30])
    parser.add_argument("--samples-per-condition", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-transfer-mib", type=int, default=150)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.max_transfer_mib <= 0:
        parser.error("--max-transfer-mib must be positive.")
    try:
        if args.inspect_only:
            _inspect_source(args.max_transfer_mib)
            return
        if args.output is None:
            parser.error("--output is required unless --inspect-only is used.")
        output = args.output.resolve()
        provenance = _create_subset(
            output, args.snrs, args.samples_per_condition, args.seed, args.max_transfer_mib
        )
        provenance_path = output.with_suffix(".provenance.json")
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        license_path = output.with_name("LICENSE.RadioML.txt")
        response = requests.get(LICENSE_URL, timeout=(15, 60))
        response.raise_for_status()
        license_path.write_bytes(response.content)
        print(json.dumps(provenance, indent=2), flush=True)
        print(f"Saved provenance to {provenance_path}", flush=True)
        print(f"Saved dataset license to {license_path}", flush=True)
    except (FileExistsError, OSError, ValueError, requests.RequestException) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

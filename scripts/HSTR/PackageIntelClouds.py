"""Package only the compiled Intel cloud libraries and their redistribution notices."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'IntelSea/releases')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'source': 'https://dpel.aswf.io/intel-cloud-library/', 'sets': {}}
    libraries = {}
    for resolution in ('full', 'half'):
        directory = ROOT / 'IntelSea' / resolution
        paths = sorted(directory.glob('*.hstrlib'))
        expected = {f'intelCloudLib_dense.{i}.L.hstrlib' for i in range(5)}
        if {p.name for p in paths} != expected or list(directory.glob('*.partial')):
            raise ValueError(f'Expected exactly five completed clouds in {directory}')
        entries = []
        for path in paths:
            with path.open('rb') as stream:
                header = stream.read(56)
                magic, version, count, table, size, source_size, rate, error, key = struct.unpack('<QIIQQQffQ', header)
                if magic != 0x3442494C52545348 or version != 2 or count != 1 or size != path.stat().st_size:
                    raise ValueError(f'Invalid package header: {path}')
                stream.seek(table + 76)
                x, y, z, voxel = struct.unpack('<IIIf', stream.read(16))
                stream.seek(table + 120)
                density_hash = struct.unpack('<Q', stream.read(8))[0]
            entries.append(dict(file=path.name, bytes=size, sha256=digest(path),
                                padded_dimensions=[x, y, z], voxel_world=voxel,
                                density_hash=f'{density_hash:016x}', lambda_value=rate,
                                compression_transmittance_error_p99=error))
        if len({e['density_hash'] for e in entries}) != 5:
            raise ValueError(f'Duplicate cloud content in {directory}')
        libraries[resolution] = paths
        manifest['sets'][resolution] = entries
    for full, half in zip(manifest['sets']['full'], manifest['sets']['half']):
        if half['voxel_world'] != full['voxel_world'] * 2:
            raise ValueError('Half-resolution voxel size must be double the full-resolution size')
    manifest_path = ROOT / 'IntelSea/manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    sums = []
    for resolution, paths in libraries.items():
        archive = args.output / f'IntelClouds-{resolution.title()}.zip'
        # The payloads are already GDeflate-compressed; ZIP_STORED avoids recompressing them.
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
            for path in paths + [ROOT / 'IntelSea/README.md', ROOT / 'IntelSea/LICENSE.txt', manifest_path]:
                z.write(path, path.relative_to(ROOT).as_posix())
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError(f'Archive CRC check failed: {archive}')
        sums.append(f'{digest(archive)}  {archive.name}')
        print(f'Verified {archive.name}: {archive.stat().st_size} bytes, five clouds plus documentation')
    (args.output / 'SHA256SUMS.txt').write_text('\n'.join(sums) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

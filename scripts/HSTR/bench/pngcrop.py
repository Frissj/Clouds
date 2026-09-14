"""Pure-Python PNG crop / difference tool (8-bit RGB or RGBA, non-interlaced).

usage: pngcrop.py out.png x y w h scale a.png [b.png]
Crops [x, x + w) x [y, y + h) from a.png. With b.png, writes |a - b| * scale instead.
"""
import struct
import sys
import zlib


def read_png(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, b""
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", body)
            assert depth == 8 and interlace == 0 and color in (2, 6)
            channels = 3 if color == 2 else 4
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = width * channels
    rows, previous, pos = [], bytearray(stride), 0
    for _ in range(height):
        kind = raw[pos]
        line = bytearray(raw[pos + 1:pos + 1 + stride])
        pos += 1 + stride
        if kind == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 255
        elif kind == 2:
            line = bytearray((a + b) & 255 for a, b in zip(line, previous))
        elif kind == 3:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + previous[i]) >> 1)) & 255
        elif kind == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = previous[i]
                c = previous[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else (b if pb <= pc else c))) & 255
        rows.append(line)
        previous = line
    return width, height, channels, rows


def write_png(path, width, height, rows):
    raw = b"".join(b"\x00" + bytes(r) for r in rows)

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
                chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


args = [v for v in sys.argv[1:] if not v.startswith("--down=")]
down = int(next((v[7:] for v in sys.argv[1:] if v.startswith("--down=")), "1"))  # Box-average k x k pixels.
out, x, y, w, h, scale = args[0], *map(int, args[1:5]), float(args[5])
_, _, ca, a = read_png(args[6])
b = read_png(args[7])[3] if len(args) > 7 else None
cb = read_png(args[7])[2] if b is not None else 0


def pixel(image, channels, i, j):
    total = [0, 0, 0]
    for jj in range(j, j + down):
        for ii in range(i, i + down):
            p = image[jj][ii * channels:ii * channels + 3]
            total = [t + v for t, v in zip(total, p)]
    return [t / (down * down) for t in total]


rows = []
for j in range(y, y + h, down):
    row = bytearray()
    for i in range(x, x + w, down):
        pa = pixel(a, ca, i, j)
        if b is None:
            row += bytes(int(v + 0.5) for v in pa)
        else:
            pb = pixel(b, cb, i, j)
            row += bytes(min(255, int(abs(p - q) * scale)) for p, q in zip(pa, pb))
    rows.append(row)
write_png(out, w // down, h // down, rows)

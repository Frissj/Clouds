# CPU samples of an nsys_export.py sqlite (run_sea.py --nsys with HSTR_NSYS_SAMPLE=1), symbolised here: nsys left every frame
# unresolved. A module's load address comes from its own unwind table (.pdata): at the right base every return address of the
# call chains falls inside a function's range. dbghelp then names the addresses from the PDB beside the module.
# Usage: python scripts/HSTR/bench/nsys_symbols.py <report>.sqlite [threads=4] [rows=40]
import bisect
import ctypes
import ctypes.wintypes as wt
import os
import sqlite3
import struct
import sys
from collections import Counter, defaultdict

db = sqlite3.connect(sys.argv[1])
thread_count = int(sys.argv[2]) if len(sys.argv) > 2 else 4
rows = int(sys.argv[3]) if len(sys.argv) > 3 else 40
names = dict(db.execute("select id, value from StringIds"))

chains = defaultdict(list)  # chain id -> [(module path, ip)], leaf first
for cid, module, ip in db.execute("select id, module, originalIP from SAMPLING_CALLCHAINS order by id, stackDepth"):
    chains[cid].append((names.get(module, "?"), ip))
samples = db.execute("select id, globalTid from COMPOSITE_EVENTS").fetchall()


def pe_info(path):
    """SizeOfImage and the sorted function ranges (RVAs) of the module's exception directory."""
    data = open(path, "rb").read()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    sections = struct.unpack_from("<H", data, pe + 6)[0]
    optional = pe + 24
    size_of_image = struct.unpack_from("<I", data, optional + 56)[0]
    pdata_rva, pdata_size = struct.unpack_from("<II", data, optional + 112 + 3 * 8)
    table = optional + struct.unpack_from("<H", data, pe + 20)[0]
    offset = None
    for s in range(sections):
        va, raw_size, raw = struct.unpack_from("<III", data, table + 40 * s + 12)
        if va <= pdata_rva < va + max(raw_size, 1):
            offset = raw + pdata_rva - va
    ranges = sorted(struct.unpack_from("<II", data, offset + 12 * i) for i in range(pdata_size // 12)) if offset else []
    return size_of_image, ranges


def find_base(ips, size_of_image, ranges):
    starts = [r[0] for r in ranges]
    lo, hi = min(ips), max(ips)
    best = (-1.0, 0)
    base = ((hi - size_of_image) // 0x10000) * 0x10000
    while base <= lo:
        inside = 0
        for ip in ips:
            rva = ip - base
            i = bisect.bisect_right(starts, rva) - 1
            inside += i >= 0 and rva < ranges[i][1]
        best = max(best, (inside / len(ips), base))
        base += 0x10000
    return best


class SYMBOL_INFOW(ctypes.Structure):
    _fields_ = [("SizeOfStruct", wt.ULONG), ("TypeIndex", wt.ULONG), ("Reserved", ctypes.c_uint64 * 2), ("Index", wt.ULONG),
                ("Size", wt.ULONG), ("ModBase", ctypes.c_uint64), ("Flags", wt.ULONG), ("Value", ctypes.c_uint64),
                ("Address", ctypes.c_uint64), ("Register", wt.ULONG), ("Scope", wt.ULONG), ("Tag", wt.ULONG),
                ("NameLen", wt.ULONG), ("MaxNameLen", wt.ULONG), ("Name", wt.WCHAR * 1024)]


dbghelp = ctypes.WinDLL("dbghelp.dll")
process = wt.HANDLE(0x7FFF)  # Any unique value: dbghelp only keys its session on it.
dbghelp.SymSetOptions(0x2 | 0x4 | 0x40)  # SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_ANYTHING
dbghelp.SymLoadModuleExW.restype = ctypes.c_uint64
dbghelp.SymLoadModuleExW.argtypes = [wt.HANDLE, wt.HANDLE, wt.LPCWSTR, wt.LPCWSTR, ctypes.c_uint64, wt.DWORD, ctypes.c_void_p, wt.DWORD]
dbghelp.SymFromAddrW.argtypes = [wt.HANDLE, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(SYMBOL_INFOW)]

# Modules with a PDB beside them: the project's own (HSTRCloud.dll, Falcor.dll, Mogwai.exe, ...).
module_ips = defaultdict(lambda: ([], []))  # path -> (return addresses, leaf addresses)
for chain in chains.values():
    for depth, (path, ip) in enumerate(chain):
        if ip:
            module_ips[path][depth == 0].append(ip)
# The PDBs lie beside their modules.
search = ";".join(sorted({os.path.dirname(p) for p in module_ips if os.path.exists(p)}))
dbghelp.SymInitializeW(process, search, False)
bases = {}
for path, (returns, leaves) in module_ips.items():
    if not path.lower().endswith((".dll", ".exe")) or not os.path.exists(path) or not os.path.exists(os.path.splitext(path)[0] + ".pdb"):
        continue
    size_of_image, ranges = pe_info(path)
    probe = sorted(set(returns or leaves))
    if not ranges or not probe:
        continue
    share, base = find_base(probe[:20000], size_of_image, ranges)
    if share < 0.9:
        print(f"no base for {os.path.basename(path)} (best {share:.2f})")
        continue
    if dbghelp.SymLoadModuleExW(process, None, path, None, base, size_of_image, None, 0):
        bases[path] = (base, base + size_of_image)
        print(f"{os.path.basename(path)}: base {base:#x}, {share:.3f} of return addresses inside functions")

cache = {}


def name_of(path, ip):
    if path not in bases or not (bases[path][0] <= ip < bases[path][1]):
        return os.path.basename(path)
    if ip not in cache:
        info = SYMBOL_INFOW()
        info.SizeOfStruct = 88  # sizeof(SYMBOL_INFOW) with its one-character name
        info.MaxNameLen = 1024
        disp = ctypes.c_uint64()
        cache[ip] = info.Name if dbghelp.SymFromAddrW(process, ip, ctypes.byref(disp), ctypes.byref(info)) else f"{os.path.basename(path)}+?"
    return cache[ip]


by_thread = defaultdict(list)
for cid, tid in samples:
    by_thread[tid].append(cid)
total = len(samples)
# The sampling covers the whole system: the threads that ran the symbolised modules, busiest first.
ours = {tid: ids for tid, ids in by_thread.items() if any(p in bases for cid in ids[:2000] for p, _ in chains.get(cid, []))}
for tid, ids in sorted(ours.items(), key=lambda kv: -len(kv[1]))[:thread_count]:
    self_count, incl = Counter(), Counter()
    for cid in ids:
        frames = [name_of(p, ip) for p, ip in chains.get(cid, [])]
        if frames:
            self_count[frames[0]] += 1
        for f in set(frames):
            incl[f] += 1
    print(f"\n== thread {tid}: {len(ids)} samples ({100 * len(ids) / total:.1f}%)")
    print("-- inclusive")
    for f, n in incl.most_common(rows):
        print(f"{100 * n / len(ids):5.1f}%  {f[:150]}")
    print("-- self")
    for f, n in self_count.most_common(rows // 2):
        print(f"{100 * n / len(ids):5.1f}%  {f[:150]}")

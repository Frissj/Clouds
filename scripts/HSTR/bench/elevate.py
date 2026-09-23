import ctypes
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path

# nsys.exe is flagged "run as administrator" (GPU metrics need it), and Windows starts such an exe only through ShellExecute:
# subprocess / CreateProcess fails with WinError 740. elevate() re-runs the calling script elevated, under pythonw (a console
# program started through ShellExecute gets a console window whatever nShow says - Windows Terminal ignores SW_HIDE - and pythonw
# has none), waits for it and exits; in the elevated copy it returns and everything the script starts inherits the elevation.
NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.1\target-windows-x64\nsys.exe"


class _ShellExecuteInfo(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("fMask", ctypes.c_ulong), ("hwnd", wintypes.HWND), ("lpVerb", wintypes.LPCWSTR),
                ("lpFile", wintypes.LPCWSTR), ("lpParameters", wintypes.LPCWSTR), ("lpDirectory", wintypes.LPCWSTR),
                ("nShow", ctypes.c_int), ("hInstApp", wintypes.HINSTANCE), ("lpIDList", ctypes.c_void_p),
                ("lpClass", wintypes.LPCWSTR), ("hkeyClass", wintypes.HKEY), ("dwHotKey", wintypes.DWORD),
                ("hIcon", wintypes.HANDLE), ("hProcess", wintypes.HANDLE)]


def elevate(script, directory):
    if ctypes.windll.shell32.IsUserAnAdmin():
        return
    info = _ShellExecuteInfo(cbSize=ctypes.sizeof(_ShellExecuteInfo), fMask=0x40, lpVerb="runas",  # SEE_MASK_NOCLOSEPROCESS
                             lpFile=str(Path(sys.executable).with_name("pythonw.exe")),
                             lpParameters=subprocess.list2cmdline([str(Path(script).resolve())] + sys.argv[1:]),
                             lpDirectory=str(directory), nShow=0)
    if not ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(info)):
        sys.exit("elevation was refused")
    ctypes.windll.kernel32.WaitForSingleObject(info.hProcess, 0xFFFFFFFF)
    sys.exit(0)


def run_hidden(command, **kwargs):
    """subprocess.run without a console window (the elevated copy has no console to share)."""
    return subprocess.run(command, creationflags=subprocess.CREATE_NO_WINDOW, **kwargs)

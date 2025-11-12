# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['filebeep_advanced_v2.py'],
    pathex=[],
    binaries=[],
    datas=[('encoder.py', '.'), ('decoder.py', '.'), ('modem.py', '.'), ('utils/compression.py', '.'), ('hellschreiber.py', '.'), ('fec.py', '.'), ('config.py', '.')],
    hiddenimports=['pygame', 'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtWidgets', 'PyQt5.QtGui', 'sounddevice', 'soundfile', 'numpy', 'scipy', 'scipy.signal', 'scipy.fft', 'sklearn', 'sklearn.ensemble', 'sklearn.tree', 'sklearn.base', 'psutil', 'pyqtgraph', 'threading', 'struct', 'binascii', 'hashlib', 'math', 'time', 'os', 'sys'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FileBeepAdvanced',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

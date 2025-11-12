# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['filebeep_advanced_v2.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('modem.py', '.'),
        ('encoder.py', '.'),
        ('decoder.py', '.'),
        ('config.py', '.'),
        ('utils/compression.py', 'utils'),
        ('received', 'received'),
        ('cache', 'cache')
    ],
    hiddenimports=[
        'sklearn.ensemble',
        'sklearn.utils._weight_vector',
        'sklearn.tree',
        'scipy.signal',
        'scipy.fft',
        'PIL',
        'PIL._imaging',
        'numpy',
        'sounddevice',
        'soundfile',
        'pyqtgraph',
        'psutil'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
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
    icon='icon.ico',
)
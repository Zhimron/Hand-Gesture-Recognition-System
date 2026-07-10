# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = [('models\\hand_landmarker.task', 'models'), ('config\\keyboard_mappings.json', 'config')]
datas += collect_data_files('mediapipe')


a = Analysis(
    ['app\\desktop_app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['cv2', 'mediapipe'],
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
    name='HandGestureRecognition',
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

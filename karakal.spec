# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['plugins\\karakal\\src\\karakal\\__main__.py'],
    pathex=['plugins/karakal/src'],
    binaries=[],
    datas=[
        ('plugins\\karakal\\src\\karakal\\resources\\icons\\karakal_light.ico', 'karakal/resources/icons'),
        ('plugins\\karakal\\src\\karakal\\resources\\icons\\karakal_light.png', 'karakal/resources/icons'),
        ('plugins\\karakal\\src\\karakal\\resources\\icons\\karakal.ico', 'karakal/resources/icons'),
        ('plugins\\karakal\\src\\karakal\\resources\\icons\\karakal.png', 'karakal/resources/icons'),
        ('plugins\\karakal\\resources\\update_client.json', 'resources'),
    ],
    hiddenimports=[],
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
    [],
    exclude_binaries=True,
    name='karakal',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='plugins\\karakal\\src\\karakal\\resources\\icons\\karakal_light.ico',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='karakal',
)

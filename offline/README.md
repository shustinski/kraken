# Kraken offline kit (Windows x64)

`offline/` creates a reproducible, air-gapped transfer kit. It deliberately
does not copy `.venv`, UV's developer cache, `target/`, build output, models,
or user data. Those paths are machine-specific or outside source control.

## Prepare the internet-connected source PC

Use a clean, committed repository. Install the exact tool versions that will
be included in the kit, then stage their offline installers/layout in a
separate directory. The source script copies that directory without guessing
download URLs or silently selecting newer releases.

The toolchain directory must contain these items and a completed
`toolchain-manifest.json` (start from `offline/toolchain-manifest.example.json`):

- CPython **3.14.2 x64** installer;
- UV **0.10.2** Windows executable or installer;
- Git for Windows and Git LFS installers;
- Rust toolchain installer matching the installed `cargo --version`;
- a Visual Studio Build Tools offline layout with
  `Microsoft.VisualStudio.Workload.VCTools`, recommended components, and a
  Windows SDK;
- an Inno Setup 6 offline installer.

Rust must be installed before creating a kit, because `Create-OfflineKit.ps1`
uses `cargo vendor --locked` to vendor the exact `blob_gateway/Cargo.lock`
graph. The script stops rather than generating a kit with missing Rust
dependencies.

```powershell
pwsh .\offline\Create-OfflineKit.ps1 `
  -OutputPath E:\Kraken-offline-kit-2026-09-16 `
  -ToolchainDirectory E:\offline-toolchains
```

The command creates a complete Git bundle of every local branch and tag, a
hash-verified Windows x64 wheelhouse derived directly from `uv.lock`, a fresh
UV cache populated with `uv sync --frozen --all-packages --all-extras
--all-groups`, the exact requirements export, Cargo vendor data, and a SHA-256
manifest. It requires network access only on this source PC while dependencies
are populated.

## Install on the offline PC

First install the bundled CPython, UV, Git/Git LFS, Rust, Visual Studio Build
Tools/SDK and Inno Setup from `toolchains/`. Their installers may require an
administrator depending on local policy. Then run:

```powershell
pwsh .\offline\Install-OfflineKit.ps1 `
  -KitPath E:\Kraken-offline-kit-2026-09-16 `
  -DestinationPath D:\code\kraken
```

The installer verifies every manifest hash, clones from the bundle, runs
`git fsck`, confirms the recorded commit, and creates `.venv` with
`uv sync --offline --frozen`. It never contacts a package index and disables
automatic Python downloads.

For Rust builds, point `CARGO_HOME` to the vendor configuration shipped in the
kit before calling the normal Windows packaging script:

```powershell
$env:CARGO_HOME = "E:\Kraken-offline-kit-2026-09-16\dependencies\cargo"
```

The current repository build script invokes Cargo with `--locked`; use
`cargo build --locked --offline --manifest-path blob_gateway\Cargo.toml` once
before packaging to prove the vendor setup. The kit does not alter global Cargo
configuration automatically.

## Exchange committed changes by USB

On the sending PC, use the last common commit (shown by `git merge-base`):

```powershell
pwsh .\offline\Export-GitUpdate.ps1 -Since <common-commit> -OutputPath E:\updates\alice-to-bob.bundle
```

On the receiving PC:

```powershell
pwsh .\offline\Import-GitUpdate.ps1 -BundlePath E:\updates\alice-to-bob.bundle
git branch -r
git merge offline-bundle/<branch>
```

The bundle declares `<common-commit>` as a prerequisite. Import therefore
fails before changing refs if the wrong base is used. Imported branches stay
under `offline-bundle/`; merge or rebase explicitly. Do not transfer a working
folder over another working folder. Commit first, and transfer uncommitted work
only through an explicitly reviewed patch.

If Git LFS files are introduced later, `Create-OfflineKit.ps1` archives local
LFS objects and lists them in the manifest. The current repository has no LFS
objects.

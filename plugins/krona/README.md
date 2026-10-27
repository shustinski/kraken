# Krona

Krona is a Kraken plugin for EDIF parsing, netlist viewing, logic extraction,
and structural analysis.

## Explicit connectivity contract

Logic extraction and the structural EDIF adapter use **only explicit
`net / joined / portRef` membership** for electrical edges. Each declared net keeps
its identity. Coincident wire endpoints, nearby segments, symbol positions, and
similarly numbered names never merge nets or fill missing pins. A pin appearing in
two different nets is a conflict, not evidence that those nets are aliases. Duplicate
net identities, missing instance references and incomplete transistor terminals are
reported; affected devices are excluded from electrical analysis. Ambiguous `G` /
`GATE` aliases also remain a conflict instead of selecting a preferred input.

Top-level port directions come from the view interface and their connection from
`portRef` without `instanceRef`. The legacy `INn` / `OUTn` naming convention and
`INSINn` explicit stimulus terminals remain supported, but drawing an input/output
symbol does not connect it. Nominal rail roles use exact conventional net names or
explicit supply port names; this convention is recorded in logic metadata and does
not measure a voltage or establish physical correctness.

Unknown transistor gates do not establish conduction. Conflicting driven levels,
possible opposite drive through an unknown gate, and incomplete/unsupported device
models preserve an unknown output instead of publishing a Boolean function. A
complete explicit synthetic CMOS inverter evaluates to `!A`; the explicit synthetic
18-transistor master/slave circuit is recognized as a rising-edge D flip-flop.
These fixtures are synthetic netlists, not microscopy or recovered silicon.

The historical `e1`, `e2`, and `e129` schematic fixtures contain conflicting pin
memberships and lack an explicit joined output. Their former flip-flop/formula
results depended on geometric repair and are no longer asserted. The files remain
negative regressions: unresolved ports and concrete pin/net conflicts are reported,
and no output connection or function is invented.

Logic JSON `meta` and structural JSON `metadata` carry
`analysis_basis = "explicit_edif_joined"`,
`geometry_used_for_connectivity = false`, and
`electrical_connectivity = "unverified"`. They contain source path/SHA-256, explicit
instance/port/net provenance and structured conflicts. A changing source file causes
analysis to fail instead of publishing mixed revisions. The Qt viewer displays
conflicts, unresolved counts and the physical-verification warning in Russian and
English, including when no output can be reconstructed.

The unsafe geometric-alias mechanism is removed. Physical geometry-to-netlist
extraction from a versioned multilayer topology with reviewed conductor/contact
evidence remains unimplemented and prevents acceptance of that pipeline. This
adapter does not prove physical connectivity, cross-layer registration, or engineer
acceptance. Device classification still depends on recognized cell/pin conventions;
dynamic charge, transistor sizing and analog behavior are outside the Boolean
switch model. SPICE and Verilog structural parsers remain explicitly unimplemented.

Regression verification from the repository root:

```powershell
.venv\Scripts\python.exe -m pytest plugins/krona/tests -q
```

On Windows, run Qt tests in a dedicated Python process with
`ctypes.windll.kernel32.SetErrorMode(3)` before importing Qt/pytest to suppress native
crash dialogs. The suite checks explicit inverter and master/slave fixtures,
disconnected coincident geometry, conflicting/reordered declarations, unknown gates
and incomplete models, parser revision invalidation (including same-size/same-mtime
rewrites), source mutation rejection, standalone CLI JSON, structural provenance,
and shown Qt windows in both languages. It does not verify microscopy or silicon.

## Run

```powershell
python -m krona --help
```

## Test

```powershell
pytest
```

## Build

```powershell
.\scripts\build_windows.ps1
```

```bash
./scripts/build_linux.sh
```

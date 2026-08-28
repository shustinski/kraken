# Third-party segmentation sources

## Multi-Separator

Source: `D:\code\multi-separator`, upstream
<https://github.com/JannikIrmai/multi-separator>, commit
`437c651ddf1452452cca4cbc3c0eed2065308486`.

Contour vendors the upstream `multi-separator-algorithms/include` tree and
`src/multi_separator.cxx`. Local changes are limited to:

- modern-MSVC compatibility for the removed
  `std::allocator::const_reference` alias;
- suppressing console progress output in the GUI;
- releasing the Python GIL while the GSS/GSG solver runs;
- placing the extension in the `contour._native` namespace.

The referenced upstream checkout contains no licence file. Its use in this
private project was explicitly authorised by the project owner. Do not
redistribute this vendored source without establishing redistribution rights.

## Berkeley Segmentation Repository OWT-UCM

Source: `D:\code\BSR/grouping/lib/contours2ucm.m` and
`D:\code\BSR/grouping/source/ucm/ucm_mean_pb.cpp`.

Verbatim copies of those two upstream implementation files and their local
licence notice are included under `third_party/bsr_owt_ucm/` so the Python
port remains directly auditable against the code supplied by the project
owner.

Copyright (C) 2009-2010 Pablo Arbelaez and contributors. The source is
licensed under the GNU Affero General Public License, version 3 or later.
Contour's Python adaptation preserves the BSR oriented-watershed semantics,
dynamic mean-boundary UCM construction, deterministic merge ordering, and
the published BSR sigmoid normalisation. The private project owner explicitly
authorised this integration. Distribution must comply with the AGPL and make
the corresponding source and licence notices available.

The complete AGPL text is shipped as `third_party/BSR-AGPL-3.0.txt`.

## LibOpenCIF

Source: `D:\code\LibOpenCIF`, upstream LibOpenCIF 1.2.0 (Moises Chavez Martinez).

Contour vendors the upstream two-file distribution (`libopencif.hh`, `libopencif.cc`)
under `third_party/libopencif/` and exposes it through the `contour._native.cif_loader`
pybind11 extension for standards-compliant CIF parsing.

License: GNU General Public License v3.0 or later. See the upstream `LICENSE` file.

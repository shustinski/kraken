from __future__ import annotations

import argparse
import json
from pathlib import Path

from logic_analyzer.bootstrap import parse_to_dict, run_gui


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize or export top-level netlist from an EDIF file.")
    parser.add_argument(
        "edf_path",
        nargs="?",
        default=str(Path("test_edifs/e1_model.EDF")),
        help="Path to EDF file",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output in export mode",
    )
    parser.add_argument(
        "--json-out",
        help="Write parsed top-level netlist JSON to this file",
    )
    args = parser.parse_args()

    if args.json_out or args.pretty:
        output = parse_to_dict(args.edf_path)
        json_text = json.dumps(output, indent=2 if args.pretty else None)
        if args.json_out:
            Path(args.json_out).write_text(json_text, encoding="utf-8")
        else:
            print(json_text)
        return

    raise SystemExit(run_gui(args.edf_path))



if __name__ == "__main__":
    main()

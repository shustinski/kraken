from __future__ import annotations

import argparse
import json
from pathlib import Path

from logic_analyzer.bootstrap import logic_functions_to_dict, parse_to_dict, run_gui, structural_analysis_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize or export top-level netlist from an EDIF file.")
    parser.add_argument(
        "edf_path",
        nargs="?",
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
    parser.add_argument(
        "--logic",
        action="store_true",
        help="Extract logic function(s) from the selected EDF and print/export JSON",
    )
    parser.add_argument(
        "--structural",
        action="store_true",
        help="Run structural sequential analysis (SCC/phase/pattern classification) and print/export JSON",
    )
    args = parser.parse_args()

    if args.structural:
        if not args.edf_path:
            parser.error("edf_path is required when using --structural")
        output = structural_analysis_to_dict(args.edf_path)
        json_text = json.dumps(output, indent=2 if args.pretty or not args.json_out else None)
        if args.json_out:
            Path(args.json_out).write_text(json_text, encoding="utf-8")
        else:
            print(json_text)
        return

    if args.logic:
        if not args.edf_path:
            parser.error("edf_path is required when using --logic")
        output = logic_functions_to_dict(args.edf_path)
        json_text = json.dumps(output, indent=2 if args.pretty or not args.json_out else None)
        if args.json_out:
            Path(args.json_out).write_text(json_text, encoding="utf-8")
        else:
            print(json_text)
        return

    if args.json_out or args.pretty:
        if not args.edf_path:
            parser.error("edf_path is required when using --json-out or --pretty")
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

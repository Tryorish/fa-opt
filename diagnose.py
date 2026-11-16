"""Save BW1000 compiler resource reports and assembly for C32/C64; no GPU run."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
KERNEL = "flash_attn_v2_r64c32d64_kernel"
OPCODE = re.compile(r"^((?:[sv]|ds|buffer|flat|global|scratch)_[a-zA-Z0-9_]+)\b")


def instruction_details(body, first_line):
    """Bounded static context, with original assembly line numbers; no hot-path inference."""
    records, waits = [], Counter()
    for line_number, raw in enumerate(body.splitlines(), first_line):
        line = re.split(r"//|;", raw, maxsplit=1)[0].strip()
        match = OPCODE.match(line)
        if match or re.fullmatch(r"[.$\w]+:", line):
            records.append((line_number, line, match.group(1) if match else ""))
        if match and match.group(1).startswith("s_waitcnt"):
            waits[" ".join(line.split())] += 1

    anchors = (("last_buffer_load", "buffer_load", -1),
               ("first_mmac", "v_mmac", 0),
               ("first_exp", "v_exp", 0),
               ("first_matrix_read", "ds_read_m", 0))
    ranges, missing = [], []
    for name, prefix, which in anchors:
        hits = [i for i, (_, _, op) in enumerate(records) if op.startswith(prefix)]
        if hits:
            i = hits[which]
            ranges.append((max(0, i - 20), min(len(records), i + 21), name))
        else:
            missing.append(name)
    # Merge overlapping windows so adjacent memory/MMAC anchors print once.
    merged = []
    for start, end, name in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].append(name)
        else:
            merged.append([start, end, [name]])
    windows = [{"anchors": names,
                "lines": [f"{number:5d}: {line}" for number, line, _ in records[start:end]]}
               for start, end, names in merged]
    return {"waitcnts": dict(waits), "windows": windows, "missing_anchors": missing}


def redact_url_credentials(text):
    # Some compiler builds embed authenticated repository URLs in --version.
    return re.sub(r"(https?://)[^\s/@]+@", r"\1[redacted]@", text)


def summarize_assembly(source, symbol=None, *, details=False):
    pattern = re.escape(symbol) if symbol is not None else r"[^\s:]*" + KERNEL + r"[^\s:]*"
    match = re.search(r"^(" + pattern + r"):", source, re.MULTILINE)
    if not match:
        return {"warning": "Target kernel label not found; inspect the saved assembly."}
    symbol = match.group(1)
    remainder = source[match.end():]
    end = re.search(r"^\.Lfunc_end\d+:|^\s*\.size\s+" + re.escape(symbol) + r"\s*,",
                    remainder, re.MULTILINE)
    if not end:
        return {"symbol": symbol, "warning": "Kernel end not found; opcode counts omitted."}
    body = remainder[:end.start()]
    # Counts describe static assembly text, not dynamic executions or cycles.
    opcodes = Counter(re.findall(r"^\s*((?:[sv]|ds|buffer|flat|global|scratch)_[a-zA-Z0-9_]+)\b",
                                 body, re.MULTILINE))
    descriptor = re.search(r"\.amdhsa_kernel\s+" + re.escape(symbol) +
                           r"\s*\n(.*?)\.end_amdhsa_kernel", source, re.DOTALL)
    resources = [] if not descriptor else [line.strip() for line in descriptor.group(1).splitlines()
                if re.search(r"vgpr|sgpr|agpr|group_segment|private_segment|wavefront", line)]
    info = {"symbol": symbol, "resource_descriptor": resources,
            "static_opcodes": dict(sorted(opcodes.items()))}
    if details:
        info.update(instruction_details(body, source[:match.end()].count("\n") + 1))
    return info


def summarize_assemblies(source, *, details=False):
    symbols = re.findall(r"^([^\s:]*" + KERNEL + r"[^\s:]*):", source, re.MULTILINE)
    return ([summarize_assembly(source, symbol, details=details) for symbol in symbols]
            if symbols else [summarize_assembly(source, details=details)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", nargs="+", type=int, choices=[32, 64], default=[32, 64])
    parser.add_argument("--details", action="store_true",
                        help="print all opcode counts, wait operands and bounded instruction context")
    parser.add_argument("--full-only", action="store_true",
                        help="print assembly summaries only for the FullKV=true specialization")
    args = parser.parse_args()
    compiler = os.environ.get("HIPCC") or shutil.which("aicc") or shutil.which("hipcc")
    if not compiler:
        parser.error("DTK compiler not found; put aicc/hipcc on PATH or set HIPCC")
    output = ROOT / "build" / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    # Keep every run isolated so old .s files cannot contaminate the report.
    run = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    version = subprocess.run([compiler, "--version"], text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=True).stdout.strip()
    version = redact_url_credentials(version)
    extra = shlex.split(os.environ.get("FA_BW_CXXFLAGS", ""))
    report = {"compiler": version, "directory": str(run), "configurations": []}
    summary = [version, f"Artifacts: {run}", "Static opcode counts are not runtime cycle counts."]
    failed = False
    for tile in args.tiles:
        directory = run / f"c{tile}"
        directory.mkdir()
        command = [compiler, "-x", "hip", "-O3", "-std=c++17", "--offload-arch=gfx936",
                   "-fPIC", "-shared", *extra,
                   "-UFA_BW_KV_TILE", f"-DFA_BW_KV_TILE={tile}",
                   "-UFA_BW_NATIVE_EXP2", "-DFA_BW_NATIVE_EXP2=1",
                   "-save-temps=obj", "-Rpass-analysis=kernel-resource-usage",
                   str(ROOT / "flash_attn_bw.hip"), "-o", str(directory / "kernel.so")]
        print(f"Compiling C{tile}; artifacts in {directory}", flush=True)
        result = subprocess.run(command, cwd=directory, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        result.stdout = redact_url_credentials(result.stdout)
        (directory / "compile.log").write_text(shlex.join(command) + "\n\n" + result.stdout)
        entry = {"kv_tile": tile, "command": command, "returncode": result.returncode,
                 "compiler_report": result.stdout, "assembly": {}}
        section = [f"\nC{tile}", shlex.join(command), result.stdout.strip()]
        if result.returncode:
            failed = True
            section.append("Compilation failed; see compile.log. No kernel changes were made.")
        else:
            for path in sorted(directory.rglob("*.s")):
                source = path.read_text(errors="replace")
                if not any(marker in source for marker in (".amdgcn_target", ".amdhsa_kernel", ".amdgpu_metadata")):
                    continue
                infos = summarize_assemblies(source, details=args.details)
                entry["assembly"][str(path)] = infos
                selected = [info for info in infos if not args.full_only or
                            re.search(re.escape(KERNEL) + r"ILb1E", info.get("symbol", ""))]
                if not selected:
                    section.append(f"Full-KV kernel not found in {path}; no summary inferred.")
                    failed = True
                for info in selected:
                    section.extend([str(path), info.get("symbol", "<unknown kernel>"),
                                    *info.get("resource_descriptor", [])])
                    if "warning" in info:
                        section.append(info["warning"])
                        if args.details:
                            failed = True
                    interesting = {op: count for op, count in info.get("static_opcodes", {}).items()
                                   if op.startswith(("v_mmac", "v_exp", "ds_", "scratch_", "s_wait", "s_barrier",
                                                     "buffer_", "global_", "flat_", "v_perm", "v_mov",
                                                     "v_cmp", "v_cndmask"))}
                    if args.details:
                        section.append("Static opcodes (all): " + json.dumps(info.get("static_opcodes", {})))
                        section.append("Static waitcnts: " + json.dumps(info.get("waitcnts", {})))
                        for window in info.get("windows", []):
                            section.append("Assembly context: " + ", ".join(window["anchors"]))
                            section.extend(window["lines"])
                        if info.get("missing_anchors"):
                            section.append("Instruction anchors not found: " + ", ".join(info["missing_anchors"]))
                    else:
                        section.append("Static opcodes (selected): " + json.dumps(interesting))
            if not entry["assembly"]:
                section.append("No AMDGPU .s file identified. All compiler artifacts were kept for inspection.")
                if args.details:
                    failed = True
        report["configurations"].append(entry)
        summary.extend(section)
    text = "\n".join(summary) + "\n"
    (run / "summary.txt").write_text(text)
    (run / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "summary.txt").write_text(text)
    print(text)
    print(f"Summary: {output / 'summary.txt'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

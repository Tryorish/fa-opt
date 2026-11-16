"""Print existing hipprof PMC files/schema/rows for pasting into chat. Stdlib only."""

import argparse
import csv
import itertools
import json
from pathlib import Path
import re
import sqlite3


def display(value, max_chars):
    if isinstance(value, bytes):
        return f"<BLOB: {len(value)} bytes>"
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + f"... <truncated {len(value) - max_chars} chars; use --max-chars>"
    return value


def print_row(row, max_chars):
    print(json.dumps([display(x, max_chars) for x in row], ensure_ascii=False))


def dump_csv(path, rows, max_chars, columns=None):
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as file:
        sample = file.read(8192)
        file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        records = list(itertools.islice(csv.reader(file, dialect), rows + 2))
    indices = None
    if columns is not None and records:
        indices = [i for i, name in enumerate(records[0]) if re.search(columns, name, re.I)]
        if not indices:
            print("No matching CSV columns. Header:")
            print_row(records[0], max_chars)
            return
        print("Column filter applied; the first CSV record is treated as the header.")
    for i, row in enumerate(records[:rows + 1]):
        if indices is not None:
            if len(row) != len(records[0]):
                print(f"CSV record {i}: width mismatch; showing the raw record:")
                print_row(row, max_chars)
                continue
            row = [row[j] for j in indices]
        print(f"CSV record {i} (first record may be a header): ", end="")
        print_row(row, max_chars)
    if len(records) > rows + 1:
        print("More CSV records omitted; increase --rows if needed.")


def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'


def dump_db(path, rows, max_tables, max_chars, table_pattern):
    # mode=ro prevents accidental database creation or data/schema writes.
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        names = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        print("SQLite tables:", json.dumps(names, ensure_ascii=False))
        selected = [name for name in names if re.search(table_pattern, name, re.I)]
        selected.sort(key=lambda name: (not bool(re.search("pmc|counter|metric|kernel|dispatch|ops", name, re.I)), name))
        for name in names:
            cursor = connection.execute(f"SELECT * FROM {quote_identifier(name)} LIMIT 0")
            print(f"Columns of {name}: " + json.dumps([x[0] for x in cursor.description], ensure_ascii=False))
        for name in selected[:max_tables]:
            data = connection.execute(f"SELECT * FROM {quote_identifier(name)} LIMIT ?", (rows + 1,)).fetchall()
            print(f"Rows of {name} (raw values; units are not inferred):")
            if not data:
                print("<empty table>")
            for row in data[:rows]:
                print_row(row, max_chars)
            if len(data) > rows:
                print("More rows omitted; increase --rows if needed.")
        if len(selected) > max_tables:
            print("More table samples omitted; use --table REGEX or increase --max-tables.")
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="existing build/hipprof/run-XXXXXX directory")
    parser.add_argument("--rows", type=int, default=3, help="data rows per CSV/table")
    parser.add_argument("--max-tables", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=2000, help="maximum characters per text cell")
    parser.add_argument("--table", default=".", help="regex selecting DB tables to sample; schemas always printed")
    parser.add_argument("--columns", help="regex selecting CSV columns by the first record's header; DB output unchanged")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("report directory does not exist")
    if min(args.rows, args.max_tables, args.max_chars) < 1:
        parser.error("limits must be positive")
    try:
        re.compile(args.table)
        if args.columns is not None:
            re.compile(args.columns)
    except re.error as error:
        parser.error(f"invalid filter regex: {error}")

    files = sorted(path for path in args.directory.rglob("*") if path.is_file())
    print("Report files:")
    for path in files:
        print(f"  {path.relative_to(args.directory)} ({path.stat().st_size} bytes)")
    candidates = [p for p in files if "pmc" in p.name.lower() and p.suffix.lower() in (".csv", ".db", ".sqlite", ".sqlite3")]
    if not candidates:
        print("No PMC-named CSV/DB found. File inventory above is needed to locate the actual export.")
    for path in candidates:
        print(f"\nFile: {path.relative_to(args.directory)}")
        try:
            if path.suffix.lower() == ".csv":
                dump_csv(path, args.rows, args.max_chars, args.columns)
            else:
                dump_db(path, args.rows, args.max_tables, args.max_chars, args.table)
        except (OSError, sqlite3.Error, csv.Error) as error:
            print(f"Could not read report: {error}")
    print("\nThis is a raw preview, not a diagnosis. Missing/empty metrics are not zero utilization.")


if __name__ == "__main__":
    main()

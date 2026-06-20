from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    print("Python 3.11+ is required for tomllib.", file=sys.stderr)
    raise SystemExit(2)


ROOT = Path(__file__).resolve().parents[2]
MEMORY = ROOT / "memory"
OUT_DIR = ROOT / ".work" / "knowledge"
DB_PATH = OUT_DIR / "research_graph.sqlite"
SUMMARY_PATH = OUT_DIR / "research_graph_summary.md"
ACTIVE_ITEM_LIMIT = 40
MEASUREMENT_LIMIT = 30
PROOF_OBJECT_LIMIT = 30


def load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def merge_ledger_shards(stub_name: str, shard_dir: str, list_keys: list[str]) -> dict:
    data = load_toml(MEMORY / stub_name)
    shard_root = MEMORY / "ledgers" / shard_dir
    if not shard_root.exists():
        return data

    merged = {key: value for key, value in data.items() if key not in list_keys}
    for key in list_keys:
        merged[key] = list(data.get(key, []))

    for shard in sorted(shard_root.glob("*.toml")):
        shard_data = load_toml(shard)
        for key in list_keys:
            merged[key].extend(shard_data.get(key, []))
    return merged


def recreate_db() -> sqlite3.Connection:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        """
        create table rounds (
            id text primary key,
            date text,
            aspect text,
            status text,
            source text,
            summary text
        );
        create table findings (
            id text primary key,
            round text,
            name text,
            status text,
            aspect text,
            projection text,
            source text,
            summary text,
            burn text
        );
        create table relationships (
            src text,
            dst text,
            type text,
            note text
        );
        create table measurements (
            id text primary key,
            round text,
            kind text,
            metric text,
            value text,
            source text,
            note text
        );
        create table proof_objects (
            id text primary key,
            family text,
            status text,
            equation text,
            source text,
            requires text
        );
        """
    )
    return conn


def insert_rows(conn: sqlite3.Connection, table: str, rows: list[dict], columns: list[str]) -> None:
    placeholders = ", ".join(["?"] * len(columns))
    col_sql = ", ".join(columns)
    sql = f"insert into {table} ({col_sql}) values ({placeholders})"
    values = [[row.get(col, "") for col in columns] for row in rows]
    conn.executemany(sql, values)


def insert_relationships(conn: sqlite3.Connection, rows: list[dict]) -> None:
    values = [[row.get("from", ""), row.get("to", ""), row.get("type", ""), row.get("note", "")] for row in rows]
    conn.executemany("insert into relationships (src, dst, type, note) values (?, ?, ?, ?)", values)


def md_cell(value: object) -> str:
    text = str(value or "")
    return text.replace("\n", " ").replace("|", "\\|")


def build_summary(conn: sqlite3.Connection) -> str:
    counts = {
        "rounds": conn.execute("select count(*) from rounds").fetchone()[0],
        "findings": conn.execute("select count(*) from findings").fetchone()[0],
        "measurements": conn.execute("select count(*) from measurements").fetchone()[0],
        "proof_objects": conn.execute("select count(*) from proof_objects").fetchone()[0],
    }
    active_count = conn.execute(
        """
        select count(*)
        from findings
        where status in ('candidate', 'validated_for_gate', 'gate_hypothesis')
        """
    ).fetchone()[0]
    active = conn.execute(
        """
        select name, status, projection, source
        from findings
        where status in ('candidate', 'validated_for_gate', 'gate_hypothesis')
        order by round desc, status, name
        limit ?
        """
        ,
        (ACTIVE_ITEM_LIMIT,),
    ).fetchall()
    measurement_count = conn.execute("select count(*) from measurements").fetchone()[0]
    measurements = conn.execute(
        """
        select metric, value, kind, source
        from measurements
        order by id
        limit ?
        """
        ,
        (MEASUREMENT_LIMIT,),
    ).fetchall()
    proof_count = conn.execute("select count(*) from proof_objects").fetchone()[0]
    proofs = conn.execute(
        """
        select id, status, equation, requires
        from proof_objects
        order by id
        limit ?
        """
        ,
        (PROOF_OBJECT_LIMIT,),
    ).fetchall()

    lines: list[str] = []
    lines.append("# Research Graph Summary")
    lines.append("")
    lines.append("Generated from human-maintained ledgers. This compact file is a cache, not source truth.")
    lines.append("Use `research_graph.sqlite` or targeted ledger excerpts for full detail.")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("| item | count |")
    lines.append("|---|---:|")
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append(f"## Active Research Items (first {len(active)} of {active_count})")
    lines.append("")
    lines.append("| item | status | projection | source |")
    lines.append("|---|---|---|---|")
    for name, status, projection, source in active:
        lines.append(
            f"| {md_cell(name)} | {md_cell(status)} | {md_cell(projection)} | `{md_cell(source)}` |"
        )
    lines.append("")
    lines.append(f"## Measurements (first {len(measurements)} of {measurement_count})")
    lines.append("")
    lines.append("| metric | value | kind | source |")
    lines.append("|---|---|---|---|")
    for metric, value, kind, source in measurements:
        lines.append(f"| {md_cell(metric)} | {md_cell(value)} | {md_cell(kind)} | `{md_cell(source)}` |")
    lines.append("")
    lines.append(f"## Proof Objects (first {len(proofs)} of {proof_count})")
    lines.append("")
    lines.append("| id | status | equation | requires |")
    lines.append("|---|---|---|---|")
    for proof_id, status, equation, requires in proofs:
        lines.append(f"| {md_cell(proof_id)} | {md_cell(status)} | `{md_cell(equation)}` | {md_cell(requires)} |")
    lines.append("")
    lines.append("## Full Detail")
    lines.append("")
    lines.append("- Query `.work/knowledge/research_graph.sqlite` for full generated detail.")
    lines.append("- Read only targeted excerpts from `memory/*-ledger.toml` stubs or `memory/ledgers/...` shards when source truth is needed.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    research = merge_ledger_shards("research-ledger.toml", "research", ["rounds", "findings", "relationships"])
    measurement = merge_ledger_shards("measurement-ledger.toml", "measurement", ["measurements"])
    proof = merge_ledger_shards("proof-object-ledger.toml", "proof-object", ["proof_objects"])

    conn = recreate_db()
    with conn:
        insert_rows(conn, "rounds", research.get("rounds", []), ["id", "date", "aspect", "status", "source", "summary"])
        insert_rows(
            conn,
            "findings",
            research.get("findings", []),
            ["id", "round", "name", "status", "aspect", "projection", "source", "summary", "burn"],
        )
        insert_relationships(conn, research.get("relationships", []))
        insert_rows(
            conn,
            "measurements",
            measurement.get("measurements", []),
            ["id", "round", "kind", "metric", "value", "source", "note"],
        )
        insert_rows(
            conn,
            "proof_objects",
            proof.get("proof_objects", []),
            ["id", "family", "status", "equation", "source", "requires"],
        )

    SUMMARY_PATH.write_text(build_summary(conn), encoding="utf-8")
    conn.close()
    print(f"Wrote {DB_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

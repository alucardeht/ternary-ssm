# Knowledge Scripts

These scripts build generated research indexes from human-maintained ledgers in
`memory/`.

The generated outputs are cache/artifacts, not source of truth.

## Build

```powershell
python scripts/knowledge/build_research_index.py
```

Outputs:

- `.work/knowledge/research_graph.sqlite`
- `.work/knowledge/research_graph_summary.md`

## Source Files

- `memory/research-ledger.toml`
- `memory/measurement-ledger.toml`
- `memory/proof-object-ledger.toml`


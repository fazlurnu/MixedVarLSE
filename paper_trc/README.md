# paper_trc — experiments for the TRC paper

Fresh start (2026-10-30). Results from before this folder (`examples/ch08_*`,
`experiment/`) are not cited in the paper.

## Rules
1. **Parameters live in `config.py` only.** Scripts read it, each run copies it into its
   `run.json`, and `make_tables.py` writes the paper's parameter tables from it.
2. **Commit before every run.** A run refuses to start on uncommitted code in
   `mvlse` or `BlueSkyCDaRR` and records both commits. `--allow-dirty` is for smoke tests only.
3. **Never edit a script that has results.** Change `config.py`, commit, rerun.
4. **`STATUS.md` is generated** by `python status.py`. Do not edit it.

## Layout
| file | paper section | what |
|---|---|---|
| `config.py` | 3, 5.1–5.3 | every parameter |
| `exp0_verify.py` | 5.4 | known-truth surfaces, active loop |
| `exp1_heading.py` | 5.5, 5.7 | 6 variables, 20/40/60 kt, worst heading |
| `exp2_surface.py` | 5.5, 5.8–5.10 | 4 CNS variables, 20 kt, 2° and 180° |
| `exp3_margin.py` | 5.11 | exp2 at 180° with margin 1.05 |
| `exp4_confirm.py` | 5.12 | skeleton: boundary confirmation |
| `make_tables.py` | 3, 5 | `tables/*.tex` for the journal |
| `status.py` | — | `STATUS.md` |

Each run writes `results/<experiment>/<run>/store.jsonl` and `run.json`.

## Order
Run in the `mvlse` conda env, from this folder.

```
python exp0_verify.py                   # minutes to hours, validates the loop
python exp1_heading.py --bracket
python exp1_heading.py
python exp2_surface.py --dpsi 180 --bracket
python exp2_surface.py --dpsi 180
python exp2_surface.py --dpsi 2 --bracket
python exp2_surface.py --dpsi 2
python exp3_margin.py
python status.py
```

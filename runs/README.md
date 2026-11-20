# Run configs

- `run3/`: every config run in run 3 (`make_configs.py` generates them); timings in `docs/run_log.md`.
- `recipe/recipe.sh`: the one-shot recipe on Modal (README §3).

**Naming.** The README calls the vectors R, D and O. The configs, bank files, results and run reports keep the names
used at the time:

| README | Configs, banks, results, run reports | Bank key / file |
|---|---|---|
| R, refusal direction | H (run 1: "harmfulness", `prompt_end`), H3 (run 3) | `prompt_end/all`; `h3/6:1` in `r3_h3.pt` |
| D, deflection direction | R ("residual", run 3) | `r/6:1` in `r3_r.pt` |
| O, onset direction | D ("decision", runs 1–2) | `onset_vs_comply/all` |

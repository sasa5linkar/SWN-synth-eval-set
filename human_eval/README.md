# Human evaluation workflow

This directory stores preserved human evaluation workbooks, extracted CSV copies,
per-evaluator statistics, and agreement reports.

## Adding another evaluator

1. Copy the evaluator workbook into the next numbered folder through the analyzer:

   ```powershell
   $env:PYTHONIOENCODING='utf-8'
   & 'C:\Users\sasa5\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
     'human_eval\analyze_human_evals.py' `
     --eval 04 'C:/Users/sasa5/Downloads/<new evaluator workbook>.xlsx'
   ```

2. Regenerate pairwise and multi-evaluator agreement reports. For a fourth
   evaluator, run the pairwise comparisons that include the new label, then run
   the all-evaluator comparison:

   ```powershell
   $env:PYTHONIOENCODING='utf-8'
   & 'C:\Users\sasa5\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
     'human_eval\analyze_human_evals.py' --eval 04 'C:/Users/sasa5/Downloads/<new evaluator workbook>.xlsx' --compare 01 04
   & 'C:\Users\sasa5\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
     'human_eval\analyze_human_evals.py' --eval 04 'C:/Users/sasa5/Downloads/<new evaluator workbook>.xlsx' --compare 02 04
   & 'C:\Users\sasa5\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
     'human_eval\analyze_human_evals.py' --eval 04 'C:/Users/sasa5/Downloads/<new evaluator workbook>.xlsx' --compare 03 04
   & 'C:\Users\sasa5\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
     'human_eval\analyze_human_evals.py' --eval 04 'C:/Users/sasa5/Downloads/<new evaluator workbook>.xlsx' --compare-three 01 02 03 04
   ```

3. Check the generated `README.md`, `*_stats.json`, `disagreements_*.csv`, and
   any focused filters such as `all_three_same_correction_vs_llm.csv`.

4. Verify Serbian text before committing:

   ```powershell
   $env:PYTHONIOENCODING='utf-8'
   & 'C:\Users\sasa5\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
     'C:\Users\sasa5\.codex\skills\serbian-text-encoding\scripts\check_serbian_text.py' `
     human_eval\04\human_eval_04.xlsx human_eval\04\human_eval_04.csv `
     --expect 'tačno' --expect 'netačno'
   ```

## Agreement metrics

- Exact agreement counts identical annotator decisions.
- Cohen kappa is used for pairwise chance-corrected agreement.
- Fleiss kappa is used for multi-annotator chance-corrected agreement.
- Binary acceptable agreement collapses `tačno` + `blizu` vs `netačno`.
- Effective value agreement compares final human sentiment values after applying
  corrections from `human_*_value`.

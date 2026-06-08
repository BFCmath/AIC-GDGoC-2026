| id | task | status | notes |
| --- | --- | --- | --- |
| update-script | Update train_bc_ppo.py (expert, symmetries, safety mask, shaped reward, opponent pool, fallback integration) | done | All features implemented in scripts/participant/train_bc_ppo.py |
| update-notebook | Update colab/base.ipynb to match training script changes and handle fallback file copying | done | Successfully synchronized notebook cells with train_bc_ppo.py |
| run-verification | Run verification smoke test command | done | Smoke test passes; local match vs 4.py/Tactical/Genius gets 6/10 wins |
| document-changes | Document changes in walkthrough.md | done | Walkthrough created in brain directory |
| track-experts | Add agent/codex/3.py and agent/codex/4.py to git and push to origin | done | Pushed expert agents to remote |
| benchmark-new-agents | Benchmark the new agents and fix their runtime/loading bugs | done | Benchmarked all 5 agents; fixed bugs in deepseek 2, grok 1, and gemini 1 |
| document-benchmark | Commit and push the benchmark documentation for the new agents | done | Documented and committed benchmark results |
| benchmark-codex-6 | Benchmark Codex 6 and document its performance | done | Completed local tournament: Codex 6 got 5 wins, 0 draws |
| benchmark-codex-7 | Benchmark Codex 7 and document its performance | done | Completed local tournament: Codex 7 got 5 wins, 0 draws |
| clash-strong-agents | Run 4-way benchmark tournament among Claude 2, Codex 4, Codex 6, and Codex 7 | done | Completed clash tournament: Codex 7 wins 10/20, Codex 4 wins 6/20 |








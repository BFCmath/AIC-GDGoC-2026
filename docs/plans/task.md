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
| rl-curriculum-script | Modify train_bc_ppo.py for PPO curriculum, shaped rewards, and Codex 4/7 fallbacks | done | Modified training script for PPO curriculum, shaped rewards, and Codex 4/7 fallbacks |
| rl-curriculum-notebook | Synchronize PPO curriculum changes into colab/base.ipynb | done | Successfully synchronized 4 cells in colab/base.ipynb |
| rl-curriculum-verify | Verify curriculum PPO script via local smoke test and pytest | done | pytest passes, smoke test runs successfully |
| benchmark-codex-8-ppo-2 | Benchmark Codex 8 and hybrid_ppo_agent_2 against Codex 4, Tactical, and Genius | done | Codex 8 got 5 wins, PPO 2 got 2 wins |
| clash-codex8-7-4-claude2 | Run 4-way clash: Codex 8, Codex 7, Codex 4, Claude 2 | done | Completed |
| clash-ppo2-7-4-claude2 | Run 4-way clash: Hybrid PPO Agent 2, Codex 7, Codex 4, Claude 2 | done | Completed |
| anti-cowardice-reward | Task 1: Update Reward Shaping to Dynamic Anti-Cowardice Mode | done | Implemented dynamic reward shaping based on 8-stage curriculum |
| wiggle-tracker | Task 2: Implement WiggleTracker in the Environment Runner | not_started |  |
| win-rate-evaluator | Task 3: Implement Objective Win-Rate Evaluator & 8-Stage Curriculum | not_started |  |
| bc-initialization | Task 4: Update BC Initialization | not_started |  |













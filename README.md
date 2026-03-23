# SUPER Expert

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/SUPER-Expert)

## Description

**SUPER Expert** is an environment for evaluating agents on scientific code execution tasks from research GitHub repositories. Agents must clone repos, install dependencies, run code (often ML training/evaluation), and report specific results.

## Capabilities

- GitHub repository comprehension
- Dependency installation and environment setup
- ML model training and evaluation
- Scientific result extraction and reporting

## Compute Requirements

Agents are given a sandboxed Docker environment. Default sandbox size is 2 CPU and 4 GB RAM. Network access enabled (agents clone repos, download data, install packages). No GPU.

## Tasks

- **Test split**: 45 tasks (Expert split from allenai/super)
- Each task involves a specific GitHub repository and a query describing what to execute and report
- Domains include NLP, ML, information retrieval, and more

## Reward Structure

Continuous reward via type-aware exact match (partial credit for dicts/lists):
- **float**: |predicted - gold| < 0.01
- **str**: Exact match after whitespace stripping
- **dict**: Average of per-key matches
- **list**: Average of per-element matches
- Reward range: 0.0 to 1.0

## Data

- **Source**: [allenai/super](https://huggingface.co/datasets/allenai/super) Expert split on HuggingFace
- **GitHub repos**: Cloned by agents at runtime (part of the task, not pre-staged)

## Tools

- **`bash`**: Execute shell commands in the sandbox (clone repos, install deps, run code)
- **`submit`**: Submit a JSON answer for evaluation (terminal action, one attempt)

## Time Horizon

Multi-turn. Tasks involve cloning repos, installing dependencies, and running code. Expected: 10–50+ tool calls.

## Environment Difficulty

Hard. Tasks require understanding research codebases, resolving dependencies, and executing ML pipelines end-to-end.

## Safety

Code is executed in an isolated sandbox. Network access is enabled for cloning public GitHub repos and installing packages.

## Citations

```bibtex
@article{bogin2024super,
  title={SUPER: Evaluating Agents on Setting Up and Executing Tasks from Research Repositories},
  author={Bogin, Ben and Yang, Kejuan and Gupta, Shashank and Richardson, Kyle and Bransom, Erin and Clark, Peter and Sabharwal, Ashish and Khot, Tushar},
  journal={arXiv preprint arXiv:2409.07440},
  year={2024}
}

@article{bragg2025astabench,
  title={AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite},
  author={Bragg, Jonathan and D'Arcy, Mike and Balepur, Nishant and Bareket, Dan and Dalvi, Bhavana and Feldman, Sergey and Haddad, Dany and Hwang, Jena D. and Jansen, Peter and Kishore, Varsha and Majumder, Bodhisattwa Prasad and Naik, Aakanksha and Rahamimov, Sigal and Richardson, Kyle and Singh, Amanpreet and Surana, Harshit and Tiktinsky, Aryeh and Vasu, Rosni and Wiener, Guy and Anastasiades, Chloe and Candra, Stefan and Dunkelberger, Jason and Emery, Dan and Evans, Rob and Hamada, Malachi and Huff, Regan and Kinney, Rodney and Latzke, Matt and Lochner, Jaron and Lozano-Aguilera, Ruben and Nguyen, Cecile and Rao, Smita and Tanaka, Amber and Vlahos, Brooke and Clark, Peter and Downey, Doug and Goldberg, Yoav and Sabharwal, Ashish and Weld, Daniel S.},
  journal={arXiv preprint arXiv:2510.21652},
  year={2025}
}
```

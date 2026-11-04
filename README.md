# agent-frameworks

A hands-on tutorial for three agent frameworks, **LangGraph**, **CrewAI** and **Strands Agents**, and a
side-by-side comparison of them.

In each framework you learn the building blocks one notebook at a time, then build the same two agents:

- **ShipFast**, a customer-support agent for a made-up shop. It sorts each message, looks up orders,
  follows a refund policy, asks a person before any refund over $100, and remembers the conversation.
- **A research agent** that answers questions about a made-up company from a fixed set of articles: it
  plans, searches (at most five tool calls), drafts an answer, critiques the draft and revises it.

Everything runs on your own machine with a small open model through Ollama. There is no API key and
nothing to pay for.

## What you need

- A Mac with Apple silicon and about **16 GB of memory**. The tutorials were written on an M4 with 16 GB.
  Linux should work with the same tools, but only the Mac has been tested.
- About **7 GB of free disk**: 5.5 GB for the models, the rest for the three Python environments.
- [Homebrew](https://brew.sh), to install the two tools below.
- Time: setup takes 15 to 30 minutes, mostly the model download. Each notebook takes a few minutes to run.

## Setup

**1. Install Ollama and uv.** Ollama serves the model; uv installs Python and each tutorial's packages.

```bash
brew install ollama uv
```

**2. Start Ollama.** Leave it running in its own terminal, or start it as a background service so it also
starts at login:

```bash
ollama serve                   # in its own terminal
# or
brew services start ollama     # in the background
```

**3. Download the models.** From the repo root:

```bash
ollama pull qwen3:8b                                       # the language model, about 5 GB
ollama pull nomic-embed-text                               # a small embedding model, used by CrewAI's memory
ollama create qwen3-agents -f contracts/ollama/Modelfile   # the shared model every tutorial uses
```

`qwen3-agents` is `qwen3:8b` with its settings built in (temperature 0, a fixed seed, a 16K context), so
all three frameworks run on exactly the same model. It takes no extra disk space.

**4. Install a tutorial's environment.** Each tutorial has its own, because the frameworks' dependencies
conflict. Always run `uv` inside a tutorial folder, never at the repo root:

```bash
cd langgraph-tutorial && uv sync     # likewise in crewai-tutorial/ and strands-tutorial/
```

uv downloads a suitable Python (3.12 or 3.13) if you do not have one. You only need the tutorials you plan
to follow.

**5. Check it works.** Still in the tutorial folder:

```bash
uv run python check_setup.py
```

It checks Python, the framework, the shared contract, Ollama and the models, then makes one short call to
the model. Every line should say `ok`; anything that says `FAIL` comes with the command that fixes it.
When it ends with "All set", open `notebooks/00_introduction.ipynb`.

## Following a tutorial

Open the notebooks with the tutorial's own environment as the kernel. Either:

- **VS Code:** open the repo, open a notebook, choose *Select Kernel*, then *Python Environments*, and pick
  the `.venv` inside that tutorial's folder (for example `langgraph-tutorial/.venv`).
- **Jupyter:** from the tutorial folder, run `uv run jupyter lab notebooks/`.

Work through the notebooks in order. Every tutorial has the same nine, so you can read the same notebook
in two frameworks side by side:

| # | Notebook | What you learn |
|---|---|---|
| 00 | Introduction | what the framework is, its core idea, and a setup check |
| 01 | The model | connect to the local model, get structured output |
| 02 | Tools | wrap the shared tools, make a tool call |
| 03 | The agent | model plus instructions plus tools, and how the loop stops |
| 04 | State and memory | remember the conversation across turns |
| 05 | Orchestration | a workflow with a bounded critique loop |
| 06 | Guardrails and approval | pause before a refund over $100, then resume |
| 07 | Tracing and evaluation | count tokens and tool calls, score a small evaluation |
| 08 | Capstone | both agents, finished and evaluated |

Each notebook ends with "How the others do this", a short note on the other two frameworks. The notes in
each notebook describe what happened when we ran it, mistakes included. A small model does not always
give the same answer, so your run can differ a little.

## Running the full evaluation

Each tutorial's `run_eval.py` runs every ShipFast scenario and research question once and writes the
results to `results/<agent>/<framework>.json`. It takes about half an hour per framework on a 16 GB M4.

```bash
cd langgraph-tutorial
uv run python run_eval.py                    # both agents
uv run python run_eval.py shipfast           # one agent
uv run python run_eval.py research Q07_deployment   # a few cases, printed but not saved
```

The results in `results/` are from our runs. Running an evaluation overwrites that framework's file.

## The comparison

**`COMPARISON.md`** is the write-up: the results, what they mean, how each framework does each job, and
when to pick which. `comparison/compare.ipynb` is its evidence: it reads the results files, counts each
build's code, and re-runs the experiment behind the biggest difference. It needs only the Python standard
library, so any of the three tutorial kernels can run it.

The short version, from one run of each case with an 8B model:

| | LangGraph | CrewAI | Strands |
|---|---|---|---|
| ShipFast (8 scenarios) | 7 | 7 | 4 |
| Research (10 questions) | 10 | 9 | 9 |

Treat it as a first measurement, not a ranking. `COMPARISON.md` explains each failure, including why
Strands' ShipFast score says more about small models and tool descriptions than about Strands.

## How the repo is organised

```
contracts/            agent_contracts: the specs, data, tools, schemas, instructions and scoring
                      that all three tutorials share
langgraph-tutorial/   notebooks/, lg_agents/ (the agents), run_eval.py, its own .venv
crewai-tutorial/      notebooks/, crew_agents/, run_eval.py, its own .venv
strands-tutorial/     notebooks/, strands_agents_impl/, run_eval.py, its own .venv
results/<agent>/      <framework>.json, written by each run_eval.py
comparison/           compare.ipynb and the recorded evidence behind COMPARISON.md
COMPARISON.md         the comparison write-up
DECISIONS.md          every design decision, with the alternative that was rejected
TUTORIAL_OUTLINE.md   the notebook outline all three tutorials follow
```

A few ideas hold the comparison together:

- **One contract.** The data, tools, instructions, scoring and evaluation runner live once, in
  `contracts/`. Each tutorial installs it as an editable package and only wraps the plain Python tools in
  its framework's tool type. The agent specs are `contracts/agent_contracts/shipfast/SPEC.md` and
  `contracts/agent_contracts/research/SPEC.md`.
- **The code you read is the code that was tested.** Each tutorial's agents live in a Python package that
  both the notebooks and `run_eval.py` import.
- **Rules that must hold are enforced in code.** The $100 approval and the refund limits are checked by the
  tools and the framework, not left to the model's judgement.

To run the contract's own tests, from any tutorial folder:

```bash
uv run --with pytest pytest -q ../contracts/tests
```

## Settings you can change

Copy `.env.example` to `.env` to override the defaults. The two that matter:

- `MODEL`: a different Ollama model (the default is `qwen3-agents`). The comparison numbers only hold for
  the default.
- `OLLAMA_HOST`: if Ollama runs somewhere other than `http://localhost:11434`.

CrewAI sends anonymous usage data by default; `crew_agents` turns that off, and `.env.example` shows the
switches.

## Troubleshooting

Start with `uv run python check_setup.py` in the tutorial folder; it names most problems and their fixes.

| What you see | What to do |
|---|---|
| `Ollama is not reachable` | Start Ollama (`ollama serve` or `brew services start ollama`). |
| `Ollama has no model 'qwen3-agents'`, or the model `does not carry the shared settings` | Run `ollama create qwen3-agents -f contracts/ollama/Modelfile` from the repo root. |
| `ModuleNotFoundError` for `lg_agents`, `crew_agents`, `strands_agents_impl` or `agent_contracts` | The notebook is on the wrong kernel. Pick the `.venv` inside that tutorial's folder, or run `uv sync` there. |
| CrewAI notebook 04 fails on embeddings | Run `ollama pull nomic-embed-text`. |
| CrewAI says it cannot run inside an event loop | Use the `_async` form, such as `await crew.kickoff_async()`; the notebooks already do. |
| Very slow answers | Close other memory-hungry apps. An 8B model wants most of 16 GB while it runs. |

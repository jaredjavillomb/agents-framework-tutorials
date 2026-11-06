# agent-frameworks

A hands-on tutorial for three agent frameworks, **LangGraph**, **CrewAI** and **Strands Agents**, and a
side-by-side comparison of them.

In each framework you learn the building blocks one notebook at a time, then build the same two agents:

- **ShipFast**, a customer-support agent for a made-up shop. It sorts each message, looks up orders,
  follows a refund policy, asks a person before any refund over $100, and remembers the conversation.
- **A research agent** that answers questions about a made-up company from a fixed set of articles: it
  plans, searches (at most five tool calls), drafts an answer, critiques the draft and revises it.

Everything runs on your own Mac or Windows PC with a small open model through Ollama. There is no API key
and nothing to pay for.

## What you need

- **A Mac or a Windows PC.** The tutorials were written and run on an M4 Mac with 16 GB. Windows uses the
  same tools and commands. They were also run on Windows 11 (16 GB, an RTX 3060 with 12 GB) with the 4B
  model: setup, all three `check_setup.py` scripts, the contract tests and 25 of the 27 notebooks (see
  "Windows run" below). The 8B has not been run on Windows. Windows 10
  and Linux should work too, but have not been tried.
- **Memory, which decides the model size** (see the table below). 16 GB runs the default 8B model.
- **Disk:** about 7 GB free with the default model, less with a smaller one.
- **Time:** setup takes 15 to 30 minutes, mostly the model download. Each notebook takes a few minutes.

Not a coder? Open **`README.html`** in your browser instead: the same setup, step by step, with every
command ready to copy.

### Which model size

Every tutorial uses one model name, `qwen3-agents`. You build it once, from one of three sizes. All three
carry the same settings, so nothing else changes; only the quality of the answers does.

| Your computer | Build from | Download | Modelfile |
|---|---|---|---|
| 16 GB or more, on an Apple silicon Mac or a PC with a graphics card (8 GB or more of its own memory) | `qwen3:8b` (default) | 5.2 GB | `contracts/ollama/Modelfile` |
| 8 GB, or a PC without a graphics card, or an Intel Mac | `qwen3:4b-q4_K_M` | 2.6 GB | `contracts/ollama/Modelfile.4b` |
| older or smaller machines | `qwen3:1.7b` | 1.4 GB | `contracts/ollama/Modelfile.1.7b` |

The notes in the notebooks, the results in `results/` and the comparison all come from the 8B. With a
smaller model expect more mistakes, especially in choosing and calling tools; the 4B and 1.7B have not
been run through the tutorials yet (DECISIONS.md 20). Without a graphics card the model runs on the
processor, which works but is slow, so a smaller model is often the better trade.

## Setup

Commands are the same on Mac and Windows unless marked. On Windows, use **PowerShell**.

**1. Install Ollama and uv.** Ollama serves the model; uv installs Python and each tutorial's packages.

- **Mac**, with [Homebrew](https://brew.sh):

  ```bash
  brew install ollama uv
  ```

- **Windows:** install Ollama with the installer from [ollama.com/download](https://ollama.com/download),
  then install uv in PowerShell and open a new PowerShell window afterwards:

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

  (`winget install Ollama.Ollama` and `winget install astral-sh.uv` work too.)

**2. Start Ollama.**

- **Mac:** `brew services start ollama` starts it now and at every login (or run `ollama serve` in its
  own terminal).
- **Windows:** the installer starts it, and it starts again at login (look for the llama icon by the
  clock).

To check, open <http://localhost:11434> in a browser: it says "Ollama is running".

**3. Get the project.** It is on GitHub at
[jaredjavillomb/agents-framework-tutorials](https://github.com/jaredjavillomb/agents-framework-tutorials).
It is public, so you do not need a GitHub account. With git (on a Mac it comes with Homebrew's setup; on Windows, `winget install Git.Git`):

```bash
git clone https://github.com/jaredjavillomb/agents-framework-tutorials.git agent-frameworks
cd agent-frameworks
```

Without git, open the repo page, choose **Code > Download ZIP**, unzip it, and rename the folder inside
from `agents-framework-tutorials-main` to `agent-frameworks`. Later, `git pull` (or a fresh ZIP) brings in
updates.

**4. Download the models.** From the repo root (the `agent-frameworks` folder), with the size from the
table above. The default:

```bash
ollama pull qwen3:8b                                       # the language model, 5.2 GB
ollama pull nomic-embed-text                               # a small embedding model, used by CrewAI's memory
ollama create qwen3-agents -f contracts/ollama/Modelfile   # the shared model every tutorial uses
```

For a smaller model, pull its base instead and point `create` at its Modelfile:

```bash
ollama pull qwen3:4b-q4_K_M                                    # or: ollama pull qwen3:1.7b
ollama pull nomic-embed-text
ollama create qwen3-agents -f contracts/ollama/Modelfile.4b    # or: Modelfile.1.7b
```

`qwen3-agents` is the base model with its settings built in (temperature 0, a fixed seed, a 16K
context), so all three frameworks run on exactly the same model. It takes no extra disk space. To switch
sizes later, run `create` again with another Modelfile; it replaces the old `qwen3-agents`.

**5. Install a tutorial's environment.** Each tutorial has its own, because the frameworks' dependencies
conflict. Always run `uv` inside a tutorial folder, never at the repo root:

```bash
cd langgraph-tutorial
uv sync
```

Likewise in `crewai-tutorial` and `strands-tutorial`. uv downloads a suitable Python (3.12 or 3.13) if you
do not have one. You only need the tutorials you plan to follow.

**6. Check it works.** Still in the tutorial folder:

```bash
uv run python check_setup.py
```

It checks Python, the framework, the shared contract, Ollama and the model (and prints its size), then
makes one short call to the model. Every line should say `ok`; anything that says `FAIL` comes with the
command that fixes it. When it ends with "All set", open `notebooks/00_introduction.ipynb`.

## Following a tutorial

Open the notebooks with the tutorial's own environment as the kernel. Either:

- **Jupyter:** from the tutorial folder, run `uv run jupyter lab notebooks/`.
- **VS Code:** open the repo, open a notebook, choose *Select Kernel*, then *Python Environments*, and pick
  the `.venv` inside that tutorial's folder (for example `langgraph-tutorial/.venv`; on Windows its
  Python is `.venv\Scripts\python.exe`).

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

There is nothing to configure: every tutorial uses `qwen3-agents` on `http://localhost:11434`. For a
smaller model, rebuild `qwen3-agents` from another Modelfile (setup step 4) rather than changing a setting.

Two environment variables override the defaults, for example `OLLAMA_HOST=http://otherbox:11434 uv run
jupyter lab notebooks/` (in PowerShell, set `$env:OLLAMA_HOST = "..."` first):

- `MODEL`: a different Ollama model. The notes in the notebooks and the comparison only hold for the
  default.
- `OLLAMA_HOST`: if Ollama runs somewhere other than `http://localhost:11434`.

CrewAI sends anonymous usage data by default; `crew_agents` turns that off (set
`CREWAI_DISABLE_TELEMETRY=false` to turn it back on).

## Windows run

Run on Windows 11 Pro, 16 GB, RTX 3060 (12 GB), Ollama 0.40.14, `qwen3-agents` built from the 4B
(`Modelfile.4b`), Python 3.12.8, PowerShell. No Windows-specific problem turned up.

- `uv sync` and `check_setup.py`: ok in all three tutorials.
- Contract tests: 28 passed.
- Notebooks, run top to bottom with `jupyter nbconvert --execute`: every notebook ran except two.
  - `crewai-tutorial` notebook 02: the model answered "where is my order A1001?" in text instead of making a
    tool call, so the cell that reads the tool calls fails (`'str' object has no attribute 'function'`).
  - `strands-tutorial` notebook 06: the agent did not attempt a refund, so there was no approval interrupt
    to pause on (`'NoneType' object is not iterable`).

  Both fail every time on the 4B and are model behaviour (the model not calling a tool), not a platform
  problem. That is the likely cause, not a proven one, because the 8B was not run on Windows to compare.
- `run_eval.py`: one research question passes in all three frameworks. The full ShipFast set on LangGraph
  passed 5 of 8 on the 4B (S4, S5 and S7 failed); the 8B on the Mac passes 7 of 8. Nothing else was
  evaluated in full.

Harmless on Windows: running a notebook headlessly prints a zmq warning about the Proactor event loop and a
warning that the kernel runs over TCP without encryption.

## Troubleshooting

Start with `uv run python check_setup.py` in the tutorial folder; it names most problems and their fixes.

| What you see | What to do |
|---|---|
| `Ollama is not reachable` | Start Ollama: on a Mac `brew services start ollama`; on Windows open the Ollama app from the Start menu. |
| `Ollama has no model 'qwen3-agents'`, or the model `does not carry the shared settings` | Run `ollama create qwen3-agents -f contracts/ollama/Modelfile` from the repo root. |
| `ModuleNotFoundError` for `lg_agents`, `crew_agents`, `strands_agents_impl` or `agent_contracts` | The notebook is on the wrong kernel. Pick the `.venv` inside that tutorial's folder, or run `uv sync` there. |
| CrewAI notebook 04 fails on embeddings | Run `ollama pull nomic-embed-text`. |
| CrewAI says it cannot run inside an event loop | Use the `_async` form, such as `await crew.kickoff_async()`; the notebooks already do. |
| Very slow answers | Close other memory-hungry apps. An 8B model wants most of 16 GB while it runs. If it stays slow, switch to a smaller model (step 4). |
| `uv` or `ollama` is not recognized (Windows) | Open a new PowerShell window after installing, so it picks up the new programs. |
| `running scripts is disabled on this system` (Windows) | Use the uv install command exactly as given; its `-ExecutionPolicy ByPass` allows that one script. |

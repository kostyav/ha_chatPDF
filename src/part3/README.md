# Part 3: The Agentic Orchestrator

## Task: 

Transform the RAG flow into a "Tool" that an autonomous agent can choose to call.

## Implementation: 

Wrap your RAG logic from src/part2 into a Python function/tool.

Initialize an agent that evaluate the user's prompt: if it requires external knowledge, it calls the RAG tool. otherwise, it answers directly.

## Tools: 
Use LangChain or LangGraph (or build a native loop using tool-calling decorators).

## Deliverables:

Agent implementation code including tool definitions.

interaction trace Showing tool usage

## Implementation Notes

`src/part3/agent.py` — single file, ~190 LOC, no new dependencies (uses `openai` + `redis`).

**Approach:** two-step prompt-based routing — no LangChain required, works with any instruction-following model.

- `_trace(msg)` — prints a cyan `[trace]` line at every decision point so tool usage is visible at runtime
- `rag_query(question)` — the tool: connects to Redis, sends retrieve requests to the running Part 2 `text_indexer` and `visual_indexer` services, collects text chunks + page images, calls Ollama to generate a grounded multimodal answer
- `run_agent(question)` — the agentic loop: **Step 1** asks the LLM to classify the question as `YES` (needs documents) or `NO` (general knowledge); **Step 2** either calls `rag_query` and synthesises the result, or answers directly
- No indexing — the Part 2 Docker stack must be running and already indexed

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design, flow diagrams, and network topology.

**Prerequisite:** Part 2 Docker stack up and indexed:
```bash
cd src/part2 && docker compose up
```

Redis and Ollama are on the internal `rag-part2_rag-net` Docker network — not exposed to the host. The agent must run as a container on the same network.

## Running

```bash
cd src/part3

# Build (only needed once or after code changes)
docker compose build

# Interactive REPL
docker compose run --rm -it agent

# Single question
docker compose run --rm agent python -m src.part3.agent --question "What does Fig. 4 show?"
```

## Interaction Trace

Every query prints `[trace]` lines (cyan) showing routing decisions and tool execution. Example session:

```
Agent ready. Type a question or 'quit' to exit.

> Short description of Bhutan
  [trace] router decision: 'NO' → direct answer
  [trace] answering directly (no retrieval)

Answer: Bhutan is a landlocked country in the Eastern Himalayas …

> What does Fig. 4 show?
  [trace] router decision: 'YES' → rag_query
  [trace] tool=rag_query  corr_id=f3a1b2c4
  [trace] pushing to queues: retrieve.text.requests, retrieve.visual.requests
  [trace] text hits: 3  best_score: 0.651  threshold: 0.3
  [trace]   text  score=0.651  pdf=23870758  "## Figure 4  The training loss curves for…"
  [trace]   text  score=0.490  pdf=23870758  "Table 2 summarises the convergence behavi…"
  [trace]   text  score=0.371  pdf=23870758  "We compare the proposed model against thre…"
  [trace] visual hits: 2
  [trace]   visual  score=14.3  doc=0  page=3  has_image=yes
  [trace]   visual  score=11.8  doc=0  page=5  has_image=yes
  [trace] calling LLM (gemma3:4b) with 3 text chunks + 2 images
  [trace] synthesising final answer from RAG result

Answer: Figure 4 shows the training loss curves comparing the baseline and the
        proposed model over 100 epochs …
```

# Constrains and rules for code generation:
1. Use the code from src/part2 to create the agent. 
2. All the code (LOC) must be kept as minimal as possible. Use 3rd party libraries as much as possible to minimize the LOC.
The code must be kept in src/part3 folder. Add more subfolders to keep the layout clean and simple

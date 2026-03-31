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


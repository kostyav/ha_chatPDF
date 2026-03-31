"""Part 4: FastAPI + Server-Sent Events wrapper around the Part 3 agentic flow.

Endpoint: POST /chat  {"question": "..."}
Streams SSE events:  status | token | done
"""
from __future__ import annotations

import asyncio
import json

import pathlib

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

import src.part3.agent as agent

app = FastAPI()


class ChatRequest(BaseModel):
    question: str


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream(question: str):
    llm = AsyncOpenAI(base_url=agent.LLM_BASE_URL, api_key="none")

    yield _sse("status", "Routing question…")
    route = await llm.chat.completions.create(
        model=agent.LLM_MODEL,
        messages=[
            {"role": "system", "content": agent._ROUTER_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=5,
    )
    needs_rag = route.choices[0].message.content.strip().upper().startswith("YES")

    if needs_rag:
        yield _sse("status", "Retrieving from documents…")
        rag_result = await asyncio.to_thread(agent.rag_query, question)
        yield _sse("status", "Synthesizing answer…")
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question using the provided context."},
            {"role": "user", "content": f"Context:\n{rag_result}\n\nQuestion: {question}"},
        ]
    else:
        yield _sse("status", "Answering from general knowledge…")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]

    async with await llm.chat.completions.create(
        model=agent.LLM_MODEL, messages=messages, max_tokens=512, stream=True
    ) as stream:
        async for chunk in stream:
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if token:
                yield _sse("token", token)

    yield _sse("done", "")


_UI = (pathlib.Path(__file__).parent / "ui.html").read_text()


@app.get("/", response_class=HTMLResponse)
async def ui():
    return _UI


@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(_stream(req.question), media_type="text/event-stream")

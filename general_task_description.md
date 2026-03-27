Machine Learning Systems Engineer Assessment: The Agentic Edge Stack
Objective
To build a locally hosted, high-performance AI Assistant capable of retrieving technical information and executing logic via an agentic loop. You will demonstrate mastery over the full lifecycle: from model optimization (quantization) to serving (streaming API) and context-awareness (RAG).
Part 1: Model Serving & Deployment
Task: Deploy a local LLM to act as the "brain" of your application.
Model Options: Use google/gemma-3-1b-it, Llama-3.2-3B-Instruct, or a comparable lightweight instructor model.
Inference Engine: You must deploy the model locally using one of the following:
Ollama 
llama.cpp 
vLLM 
Deliverables:
deployment script (Shell or Docker Compose) that initializes the inference server.
verification script (Python or cURL) demonstrating a successful "Hello World" response from the model endpoint.

Part 2: Knowledge Retrieval - In-Memory RAG 
Task: Implement a lightweight RAG system to provide the model with domain-specific knowledge.
The Dataset: Create or find a small dataset to use for the RAG. This could be technical documentation, a whitepaper, or a product manual. The content should be concise, totalling roughly 2 to 10 pages of text.
Implementation:
Embeddings: Use an open-source embedding model (e.g., sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5).
Vector Store: Implement an in-memory vector store using FAISS or ChromaDB.
Flow: Create a logic gate that takes a user query, embeds it, retrieves the top 3 relevant chunks, and injects them into the LLM prompt.
Deliverables: 
The dataset used.
The complete RAG code
A test log showing a user query and the corresponding text chunks retrieved from memory.

Part 3: The Agentic Orchestrator
Task: Transform the RAG flow into a "Tool" that an autonomous agent can choose to call.
Implementation: 
Wrap your RAG logic from Part 2 into a Python function/tool.
Initialize an agent that evaluate the user's prompt: if it requires external knowledge, it calls the RAG tool. otherwise, it answers directly.
Tools: Use LangChain or LangGraph (or build a native loop using tool-calling decorators).
Deliverables
Agent implementation code including tool definitions.
interaction trace Showing tool usage

Part 4: API Serving & Streaming
Task: Wrap the agentic flow in a production-ready web interface.
Framework: Python fastapi.
Endpoint: Create a /chat endpoint.
Response Streaming: Implement real-time streaming (Server-Sent Events). The user should see the agent’s response as it is being generated, rather than waiting for the entire block of text.
Deliverable: The application code.

Part 5: Advanced Challenges (Bonuses)
1. Structured Output Responses
Task: Ensure the agent can output specific data in a strict JSON format (e.g., extracting "Topics" and "Sentiment" from the user query) without breaking the stream.
Tool: Use Pydantic for schema validation.
Deliverable: A code snippet and example output showing the model adhering to the JSON schema.
2. Model Quantization & Performance Profiling
Task: Compare the performance of your chosen model across two different quantization levels (e.g., 1-bit vs 4-bit vs 8-bit).
Metrics: Report on tokens per second (TPS) and peak VRAM/RAM usage.
Analysis: Provide a summary of the trade-off between speed and output quality.
Deliverable: A performance report (table) detailing Tokens Per Second (TPS) and Peak VRAM/RAM usage for both versions.
3. Production-Grade Vector DB on K8s
Task: Instead of an in-memory store, deploy a standalone Vector Database (e.g., Qdrant or Weaviate) on a local Kubernetes system (such as k3s, minikube, or Kind).
Deliverable: The K8s manifest files (YAML) and the updated configuration pointing to the K8s service.

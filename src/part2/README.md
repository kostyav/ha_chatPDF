# PART2 

## Task: 
This project implements a Retrieval-Augmented Generation (RAG) system designed to answer natural language questions from a subset of the VisualMRC dataset. It is specifically optimized to handle PDFs containing text, tables, and technical drawings (e.g., chemical synthesis schemes, epidemiological maps) using a T4 GPU (16GB VRAM). Implement a lightweight RAG system to provide the model with domain-specific knowledge.


## Core Stack
Parser: Docling (Layout-aware extraction of tables and figures).

Retriever: ColQwen2-2B via Byaldi (Visual-semantic embeddings).

Generator: Gemma 3 4B (Multimodal LLM for reasoning).

Inference Engine: Ollama or vLLM or llamaCPP

## Model Options: 
Use an open-source embedding model.The list is not final and new models can be added later during production

[sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5]

The Dataset: Create or find a small dataset to use for the RAG. This could be technical documentation, a whitepaper, or a product manual. The content should be concise, totalling roughly 2 to 10 pages of text.

## Deliverables: 
The dataset used.
The complete RAG code
A test log showing a user query and the corresponding text chunks retrieved from memory.


## Dataset Structure
Source: 10 PDF files in "example_data" folder

Ground Truth: A train_dataframe_subset.csv in "example_data" folder file containing question, answer, and source_pdf mappings.

This file contains NL questions and answers for PDF files. These questions and files are from open dataset: https://www.kaggle.com/competitions/pdfvqa/overview. You can find the description of this dataset in: https://arxiv.org/pdf/2304.06447.

Code Generation Instructions
1. Document Parsing (Docling)
The model must use Docling to process the PDF. Unlike standard PDF readers, Docling should be configured to:

Identify and export tables as Markdown (e.g., the antimicrobial activity tables in 23870758.pdf).

Identify figures/schemes and save them as high-resolution images for the multimodal generator.

2. Multimodal Indexing (Byaldi + ColQwen2)
The retriever must use ColQwen2-2B to index the visual layout of each page.

Why: Technical schemes (like the smallpox eradication flow in 24069913.pdf) are often not captured by text embeddings. ColQwen2 indexes the "patches" of the image.

Implementation: Use Byaldi to create a local vector store.

3. RAG Interaction Logic
For each question in the CSV:

Retrieve: Query the Byaldi index to find the top-K most relevant page images.

Context Construction: Combine the question with the retrieved page images and the Markdown text extracted by Docling.

Generation: Send the image + text context to Gemma 3 4B.

Verification: If the similarity score from Byaldi is below a defined threshold, the system must output: "The document does not contain information about this query."

4. Vector Store: Implement an in-memory vector store using FAISS or ChromaDB.

Flow: Create a logic gate that takes a user query, embeds it, retrieves the top 3 relevant chunks, and injects them into the LLM prompt.


5. GPU Optimization (T4 16GB)
To prevent Out-of-Memory (OOM) errors:

Load ColQwen2-2B in 4-bit quantization.

Run Gemma 3 4B via Ollama using the q4_K_M GGUF format (~2.5GB VRAM).

Limit image resolution to 300 DPI to balance visual clarity with memory overhead.

Example Interaction Flow
Input Question: "What is the yield percentage of compound 12 in Toluene?"

Retrieval: Byaldi identifies Table 7 on Page 6 of 23870758.pdf visually.

Reasoning: Gemma 3 analyzes the Markdown table provided in the prompt context.

Output: "The yield for compound 12 in Toluene is 92%."

## Evaluation Script Requirements
Generate a script that iterates through the CSV, compares the system-generated answer to the ground truth using a metric like BERTScore, and logs instances where visual schemes were required for the correct answer.


## Constrains and rules for code generation:
1. It is still unknown what inference engine will be used at production stage. 
The general abstraction must be created that during the init process selects one of the 3 engines. 
So each engine must be separately loaded with its corresponding config.

2. Neither the model itself not its quantization are known and must be obtained from the config file during the "deployment script" call.

3. The verification script from the "Deliverables" must be created in tests/part1 folder.

4. More tests must be created to verify different configurations of inference engine, model and quantization.

5. All the code (LOC) must be kept as minimal as possible. Use 3rd party libraries as much as possible to minimize the LOC.
The code must be kept in src/part2 folder. Add more subfolders to keep the layout clean and simple
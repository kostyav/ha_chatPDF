# PART2 

## Task: 

Implement a lightweight RAG system to provide the model with domain-specific knowledge.

### Model Options: 
Use an open-source embedding model.The list is not final and new models can be added later during production 
[sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5]

The Dataset: Create or find a small dataset to use for the RAG. This could be technical documentation, a whitepaper, or a product manual. The content should be concise, totalling roughly 2 to 10 pages of text.

# Implementation:

Vector Store: Implement an in-memory vector store using FAISS or ChromaDB.

Flow: Create a logic gate that takes a user query, embeds it, retrieves the top 3 relevant chunks, and injects them into the LLM prompt.

# Deliverables: 
The dataset used.
The complete RAG code
A test log showing a user query and the corresponding text chunks retrieved from memory.


# Constrains and rules for code generation:
1. It is still unknown what inference engine will be used at production stage. 
The general abstraction must be created that during the init process selects one of the 3 engines. 
So each engine must be separately loaded with its corresponding config.

2. Neither the model itself not its quantization are known and must be obtained from the config file during the "deployment script" call.

3. The verification script from the "Deliverables" must be created in tests/part1 folder.

4. More tests must be created to verify different configurations of inference engine, model and quantization.

5. All the code (LOC) must be kept as minimal as possible. Use 3rd party libraries as much as possible to minimize the LOC.
The code must be kept in src/part1 folder. Add more subfolders to keep the layout clean and simple
# PART1 


## Task: 

Deploy a local LLM to act as the "brain" of your application.

### Model Options: 
The list is not final and new models can be added later during production 
[google/gemma-3-1b-it, Llama-3.2-3B-Instruct, ColQwen2-2B, Gemma3-4B]

### Inference Engine: 
You must deploy the model locally using one of the following:
### [Ollama, llama.cpp, vLLM]

# Deliverables:
1. deployment script (Shell or Docker Compose) that initializes the inference server.
2. verification script (Python or cURL) demonstrating a successful "Hello World" response from the model endpoint.

# Constrains and rules for code generation:
1. It is still unknown what inference engine will be used at production stage. 
The general abstraction must be created that during the init process selects one of the 3 engines. 
So each engine must be separately loaded with its corresponding config.

2. Neither the model itself not its quantization are known and must be obtained from the config file during the "deployment script" call.

3. The verification script from the "Deliverables" must be created in tests/part1 folder.

4. More tests must be created to verify different configurations of inference engine, model and quantization.

5. All the code (LOC) must be kept as minimal as possible. Use 3rd party libraries as much as possible to minimize the LOC.
The code must be kept in src/part1 folder. Add more subfolders to keep the layout clean and simple

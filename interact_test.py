from llama_cpp import Llama
llm = Llama(model_path="/mldata2/cache/transformers/llama2/llama-2-13b-chat.Q5_K_M.gguf")
output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
print(output['choices'][0]['text'])

# /mldata2/cache/transformers/llama/hf/7B
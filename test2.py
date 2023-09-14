from llama_cpp import Llama
import time

t1 = time.time()

llm = Llama( model_path="/mldata2/cache/transformers/llama2/llama-2-13b-chat.Q5_K_M.gguf", verbose=True, n_ctx=512, n_batch= 1024, n_gpu_layers=100 )
output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
t2 = time.time()
print(t2-t1) # output = llm("Q : Among Korean dramas, please recommend 3 medical dramas about hospital life. When recommending, classify it by number and title, and describe the release year and cast.A: ", max_tokens=32, stop=["Q:", "\n"], echo=True) # t3 = time.time() #
# print(t3-t1)
print(output)
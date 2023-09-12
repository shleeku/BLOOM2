import os
import urllib.request


def download_file(file_link, filename):
    # Checks if the file already exists before downloading
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(file_link, filename)
        print("File downloaded successfully.")
    else:
        print("File already exists.")

# Dowloading GGML model from HuggingFace
ggml_model_path = "https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/raw/main/llama-2-7b-chat.Q4_K_M.gguf"
filename = "llama-2-7b-chat.Q4_K_M.gguf"

download_file(ggml_model_path, filename)
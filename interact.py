import logging
import torch
import transformers
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast
from datasets import load_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

logger.info("Initializing model")
torch.cuda.empty_cache()


data_dir="/mldata2/cache/transformers/bloom/"
checkpoint = "bigscience/bloom-7b1"
model = BloomForCausalLM.from_pretrained(checkpoint, cache_dir=data_dir)
tokenizer = BloomTokenizerFast.from_pretrained(checkpoint, cache_dir=data_dir)

with open('context.txt') as f:
    context_init = f.readlines()

dialogue_init = []

for i in context_init:
    if "<s>" in i[:3]:
        dialogue_init.append(i)

with open("dialogue_history.txt", "w") as f:
    f.write(''.join(dialogue_init))

flag = True
while flag:

    cont = input("Input: ")
    cont = "<s> " + cont + " </s>\n"
    with open('context.txt') as f:
        context_lines = f.readlines()
    with open('dialogue_history.txt') as f:
        dialogue = f.readlines()

    context = []
    for i in context_lines:
        if "<s>" not in i[:3]:
            context.append(i)

    dialogue.append(cont)
    # lines = context + dialogue
    # prompt = ''.join(lines)
    prompt_context = ''.join(context)
    prompt_dialogue = ''.join(dialogue)


    if "bye" in cont:
        flag = False
    else:
        # maximum sequence length for BLOOM is 2048 tokens
        # here we truncate older dialogue history while preserving context to keep within limit
        result_length = 2048
        # inputs = tokenizer(prompt, return_tensors="pt")
        tokens_context = tokenizer.tokenize(prompt_context)
        tokens_dialogue = tokenizer.tokenize(prompt_dialogue)
        ids_context = tokenizer.convert_tokens_to_ids(tokens_context)
        ids_dialogue = tokenizer.convert_tokens_to_ids(tokens_dialogue)
        max_len_dialogue = result_length - len(ids_context)
        if len(ids_dialogue) > max_len_dialogue:
            ids_dialogue = ids_dialogue[-max_len_dialogue:]
        ids_combined = ids_context + ids_dialogue
        input_combined = torch.tensor([ids_combined])
        # print("\n" + 100 * '-')
        # print("Output:[Greedy]\n" + 100 * '-')
        # Greedy
        # output = tokenizer.decode(model.generate(inputs["input_ids"],
        #                                       max_length=result_length,
        #                                       # max_time=4.0
        #                                       )[0])


    #     print("\n" + 100 * '-')
    #     print("Output:[Beam Search]\n" + 100 * '-')
    #     # # Beam Search
    #     output = tokenizer.decode(model.generate(inputs["input_ids"],
    #                                           max_length=result_length,
    #                                           num_beams=2,
    #                                           # no_repeat_ngram_size=2,
    #                                           # early_stopping=True,
    #                                           # max_time=6.0
    #                                           )[0])
    #     # print("\n" + 100 * '-')
    #     # print("Output:[Sampling]\n" + 100 * '-')
    #     # # Sampling Top-k + Top-p

#--------------------------------------------------------------------------------------------
        # dialogue_text = tokenizer.decode(input_combined[0])
        # print(tokenizer.decode(input_combined[0]))
        # print(input_combined)
        output = tokenizer.decode(model.generate(input_combined,
                                              max_length=result_length,
                                              do_sample=True,
                                              top_k=50,
                                              top_p=0.9,
                                              # max_time=6.0
                                              )[0])

        output_list = list(output.split("\n"))
        with open("dialogue_history.txt", "a") as f:
            f.write(cont + output_list[-1] + "\n")
        if output_list[-1][:3] == "<s>":
            output_text = output_list[-1][3:]
        else:
            output_text = output_list[-1]
        if output_list[-1][-4:] == "</s>":
            output_text = output_text[:-4]
        print("Monica:", output_text)
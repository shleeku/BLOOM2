import logging
import torch
import transformers
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def main():

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    logger.info("Initializing model")
    torch.cuda.empty_cache()

    ### BLOOM ###
    data_dir="/mldata2/cache/transformers/bloom/"
    checkpoint = "bigscience/bloom-7b1"
    model = BloomForCausalLM.from_pretrained(checkpoint, cache_dir=data_dir)
    tokenizer = BloomTokenizerFast.from_pretrained(checkpoint, cache_dir=data_dir)

    # ### VICUNA ###
    # data_dir="/mldata2/cache/transformers/vicuna/"
    # checkpoint = "lmsys/vicuna-13b-delta-v0"
    # model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=data_dir)
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=data_dir)

    with open('context.txt') as f:
        context_init = f.readlines()

    dialogue_init = []

    for i in context_init:
        if "<s>" in i[:3]:
            dialogue_init.append(i)

    with open("dialogue_history.txt", "w") as f:
        f.write(''.join(dialogue_init))

    friend = "AI chatbot"

    flag = True
    while flag:

        cont = input("Input: ")
        cont = "<s> " + cont + " </s>\n"

        if "brad" in cont.lower() and friend != "Brad":
            friend = "Brad"
            print("{}:".format(friend), "Hey it's {}!".format(friend))
            continue
        if "elise" in cont.lower() and friend != "Elise":
            friend = "Elise"
            print("{}:".format(friend), "Hey it's {}!".format(friend))
            continue
        if "jenny" in cont.lower() and friend != "Jenny":
            friend = "Jenny"
            print("{}:".format(friend), "Hey it's {}!".format(friend))
            continue
        if "jimmy" in cont.lower() and friend != "Jimmy":
            friend = "Jimmy"
            print("{}:".format(friend), "Hey it's {}!".format(friend))
            continue
        if "john" in cont.lower() and friend != "John":
            friend = "John"
            print("{}:".format(friend), "Hey it's {}!".format(friend))
            continue
        if "laura" in cont.lower() and friend != "Laura":
            friend = "Laura"
            print("{}:".format(friend), "Hey it's {}!".format(friend))
            continue
        if "lizzy" in cont.lower() and friend != "Lizzy":
            friend = "Lizzy"
            print("{}:".format(friend), "Hey it's {}!".format(friend))
            continue
        if "skye" in cont.lower() and friend != "Skye":
            friend = "Skye"
            print("{}:".format(friend), "Hey it's {}!".format(friend))
            continue
        if "tyler" in cont.lower() and friend != "Tyler":
            friend = "Tyler"
            print("{}:".format(friend), "Hey it's {}!".format(friend))
            continue

        if friend == "AI chatbot":
            context_file = "context_init.txt"
        else:
            context_file = "context_{}.txt".format(friend.lower())

        with open(context_file) as f:
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
            print("{}:".format(friend), output_text)

if __name__ == "__main__":
    main()
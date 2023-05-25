import logging
import torch
import transformers
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from User_profile_detector import extract_user_info
from User_profile_detector import save_user_history

def main():

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    logger.info("Initializing model")
    torch.cuda.empty_cache()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")

    ### BLOOM ### 560m 1b1 1b7 3b 7b1
    data_dir="/mldata2/cache/transformers/bloom/"
    # checkpoint = "bigscience/bloom-7b1"
    checkpoint = "bigscience/bloom-3b"
    model = BloomForCausalLM.from_pretrained(checkpoint, cache_dir=data_dir).to(device)
    tokenizer = BloomTokenizerFast.from_pretrained(checkpoint, cache_dir=data_dir)

    # ### VICUNA ###
    # data_dir="/mldata2/cache/transformers/vicuna/"
    # checkpoint = "lmsys/vicuna-13b-delta-v0"
    # model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=data_dir)
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=data_dir)

    with open('context_init.txt') as f:
        context_init = f.readlines()

    dialogue_init = []

    for i in context_init:
        if "<s>" in i[:3]:
            dialogue_init.append(i)

    with open("dialogue_history.txt", "w") as f:
        f.write(''.join(dialogue_init))

    # MAAB: Clear the user input history file
    with open("user_input_history_ai chatbot.txt", "w+") as f:
        f.write("")

    friend = "AI"

    flag = True
    while flag:

        breakout = False
        cont = input("Input: ")
        cont = "<s> User: " + cont + " </s>\n"

        names = ["Brad", "Elise", "Jenny", "Jimmy", "John", "Laura", "Lizzy", "Skye", "Tyler"]
        for n in names:
            if n.lower() in cont.lower() and friend != n:
                breakout = True
                friend = n
                with open('context_{}.txt'.format(friend.lower())) as f:
                    context_init = f.readlines()
                dialogue_init = []
                for i in context_init:
                    if "<s>" in i[:3]:
                        dialogue_init.append(i)
                with open("dialogue_history.txt", "w") as f:
                    f.write(''.join(dialogue_init))
                print("{}:".format(friend), "Hey it's {}!".format(friend))
        if breakout == True:
            continue

        if friend == "AI":
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
            # output = tokenizer.decode(model.generate(input_combined.to(device),
            #                                       max_length=result_length,
            #                                       # max_time=4.0
            #                                       )[0])


        #     print("\n" + 100 * '-')
        #     print("Output:[Beam Search]\n" + 100 * '-')
            # # Beam Search
            # output = tokenizer.decode(model.generate(input_combined.to(device),
            #                                       max_length=result_length,
            #                                       num_beams=2,
            #                                       # no_repeat_ngram_size=2,
            #                                       # early_stopping=True,
            #                                       # max_time=6.0
            #                                       )[0])
        #     # print("\n" + 100 * '-')
        #     # print("Output:[Sampling]\n" + 100 * '-')
        #     # # Sampling Top-k + Top-p

    #--------------------------------------------------------------------------------------------
            # dialogue_text = tokenizer.decode(input_combined[0])
            # print(tokenizer.decode(input_combined[0]))
            # print(input_combined)
            # --------------SAMPLING------------------------------------------------------------------------------
            found = False
            counter = 0
            while found == False and counter<100:
                counter += 1
                total_output = model.generate(input_combined.to(device),
                                                               max_length=result_length,
                                                               do_sample=True,
                                                               # top_k=50,
                                                               # top_p=0.8,
                                                               top_k=100,
                                                               top_p=0.70,
                                                               # max_time=6.0,
                                                               num_return_sequences=1
                                                               )

                output_sequences = []
                for sequence in total_output:
                    output_sequences.append(tokenizer.decode(sequence, skip_special_tokens=True))

                generations = []
                for i in output_sequences:
                    generations.append(list(i.split("\n"))[-1])
                #     print("LEN of sequence:", len(i))
                #     print("output:", i[:-1])
                # print("LEN of total_output:", len(total_output))
                # for i in generations:
                    # print("generation:", i)
                found = False
                for i in output_sequences:
                    sequence_list = list(i.split("\n"))
                    if sequence_list[-1].strip() != "" and not "user:" in sequence_list[-1].lower() and not sequence_list[-1].strip()=="AI:":
                        output = i
                        found = True
                        break
            if found == False:
                output = "I don't know what to say."


            # --------------SAMPLING------------------------------------------------------------------------------
            output_list = list(output.split("\n"))
            with open("dialogue_history.txt", "a") as f:
                f.write(cont + output_list[-1] + "\n")
            if output_list[-1][:3] == "<s>":
                output_text = output_list[-1][3:]
            else:
                output_text = output_list[-1]
            if output_list[-1][-4:] == "</s>":
                output_text = output_text[:-4]
            print(output_text)
            # MAAB: Save user input history
            save_user_history(friend, cont)

    #MAAB: dialogue history
    dialogue_history = './user_input_history_ai.txt' # Create an empty list to store the dialogue history
    # Now you have the contents of the file stored in the dialogue_history list
    user_info = extract_user_info(dialogue_history)

if __name__ == "__main__":
    main()


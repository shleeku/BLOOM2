import logging
import torch
import transformers
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset
from User_profile_detector import extract_user_info
from User_profile_detector import save_user_history
from huggingface_hub import login
from transformers import pipeline
from matplotlib import pyplot as plt
import pandas as pd
from User_profile_detector import write_user_info
from User_profile_detector import convert_user_info_to_sentences
from User_profile_detector import write_sentence_to_file
from llama_cpp import Llama
from transformers import LlamaForCausalLM, LlamaTokenizer



def main():

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.propagate = False

    logger.info("Initializing model")
    torch.cuda.empty_cache()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    # ### BLOOM ### 560m 1b1 1b7 3b 7b1
    # data_dir="/mldata2/cache/transformers/bloom/"
    # # checkpoint = "bigscience/bloom-7b1"
    # checkpoint = "bigscience/bloom-3b"
    # model = BloomForCausalLM.from_pretrained(checkpoint, cache_dir=data_dir, device_map="auto").to(device)
    # # model = BloomForCausalLM.from_pretrained(checkpoint, cache_dir=data_dir).to(device)
    # tokenizer = BloomTokenizerFast.from_pretrained(checkpoint, cache_dir=data_dir)


    # # ### LLAMA 2 ###
    # login()
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(device)
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto", load_in_4bit=True, torch_dtype=torch.float32)
    # tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ")
    # model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ").to(device)

    ### LLAMA 2 C++ ###
    # tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ", use_fast=True)
    # # model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ", device_map="auto")
    #
    # # model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ", device_map="auto")
    # # model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ", device_map="auto").to(device)
    # model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ").to(device)
    #
    # # model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
    # # # To use a different branch, change revision
    # # # For example: revision="gptq-4bit-64g-actorder_True"
    # # model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
    # #                                              torch_dtype=torch.float16,
    # #                                              device_map="auto",
    # #                                              revision="main").to(device)
    # # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    #
    # # model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ").to(device)
    # # for param in model.parameters():
    # #     print(param.device)
    #
    # # checkpoint = "/mldata2/cache/transformers/llama/hf/7B"
    # # model = LlamaForCausalLM.from_pretrained(checkpoint).to(device)
    # # # model.config.max_position_embeddings = 64
    # # tokenizer = LlamaTokenizer.from_pretrained(checkpoint)

    ### LLAMA 2 GGUF ###

    tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ", use_fast=True)
    # llm = Llama(model_path="/mldata2/cache/transformers/llama2/llama-2-7b-chat.Q4_K_M.gguf", verbose=True, n_gpu_layers=100)
    llm = Llama(model_path="/mldata2/cache/transformers/llama2/llama-2-7b-chat.Q4_K_M.gguf", verbose=False,
                n_ctx=2048, n_batch=1024, n_gpu_layers=100)


    # ### DOLLY ###
    # data_dir="/mldata2/cache/transformers/dolly/"
    # checkpoint = "databricks/dolly-v2-3b"
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=data_dir)
    # model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=data_dir, device_map="auto").to(device)
    # # model.config.pad_token_id = tokenizer.eos_token_id

    # ### VICUNA ###
    # data_dir="/mldata2/cache/transformers/vicuna/"
    # # checkpoint = "lmsys/vicuna-13b-delta-v0"
    # checkpoint = "lmsys/vicuna-7b-v1.3"
    # model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=data_dir).to(device)
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=data_dir)

    ## VICUNA (BLOKE) ###
    # data_dir="/mldata2/cache/transformers/vicuna/"
    # checkpoint = "TheBloke/wizardLM-7B-GPTQ"
    # model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=data_dir).to(device)
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
    with open("user_input_history_ai.txt", "w+") as f:
        f.write("")

    friend = "AI"

    id_received = False
    while not id_received:
        try:
            user_id = int(input("User ID: "))
            id_received = True
        except ValueError:
            print("Please enter an integer.")
    # user_id = int(input("User ID: "))

    flag = True
    while flag:
        breakout = False
        cont = input("Input: ")
        words = cont.split()
        greetings = ["hallo", "hullo", "lo", "ol"]
        for i in range(len(words)):
            if words[i] in greetings:
                words[i] = "hello"
        cont = " ".join(words)
        cont = "<s> User: " + cont + " </s>\n"

        names = ["Brad", "Jenny", "Jimmy", "John", "Laura", "Tyler", "Sky",  "Lizzie", "Alice"]
        friend_list = {"Brad": ["weightlifting", "drinking", "partying"],
        "Sky": ["volleyball", "skiing", "arts and crafts"],
        "Lizzie": ["movies", "reading", "going to the beach"],
        "Tyler": ["hanging out", "skateboarding", "surfing"],
        "Jenny": ["writing", "politics", "environmental issues"],
        "Jimmy": ["rock band", "singing", "philosophy"],
        "Laura": ["knitting", "skydiving", "classical music"],
        "Alice": ["ballet", "jazz music", "opera"],
        "John": ["western movies", "broadway musicals", "history"]}
        young_friends = ["Tyler", "Sky",  "Lizzie"]
        middle_friends = ["Brad", "Jenny", "Jimmy"]
        old_friends = ["Laura", "John", "Alice"]
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
            # input_combined = torch.tensor([ids_combined])
            # MAAB: Add the new sentences to the context file
            converted_sentence = []
            if converted_sentence:
                new_sentences = converted_sentence[-1]
                new_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(new_sentences))
                ids_combined += new_ids

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
            while found == False and counter<25:
                counter += 1
                # input_combined = input_combined.to(torch.float32)
                # input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()

                ### HF LLM ###
                # total_output = model.generate(
                #                             input_combined.to(device=device),
                #                               # input_combined.to(torch.float32),
                #                                                max_length=result_length,
                #                                                do_sample=True,
                #                                                # top_k=50,
                #                                                # top_p=0.8,
                #                                                top_k=100,
                #                                                top_p=0.70,
                #                                                # max_time=6.0,
                #                                                num_return_sequences=1,
                #                                                 temperature=0.8,
                #                               # attention_mask=torch.tensor([1]*len(input_combined[0])).unsqueeze(0).to(device),
                #                                                )
                #
                # output_sequences = []
                # for sequence in total_output:
                #     output_sequences.append(tokenizer.decode(sequence, skip_special_tokens=True))

                ### GGUF LLM ###
                output_sequences = []
                llm_output = llm("\n".join([prompt_context, prompt_dialogue]), max_tokens=100, stop=["Q:", "\n"],
                             echo=True)
                for i in range(len(llm_output['choices'])):
                    output_sequences.append(llm_output['choices'][i]['text'])

                # print(output['choices'][0]['text'])

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
            print(output_text.strip())

            # --------------ACTION------------------------------------------------------------------------------
            action = ""
            if "hi" in output_text.lower().split() or "hello" in output_text.lower().split() or "bye" in output_text.lower().split() or "hey" in output_text.lower().split()\
                    or "hi!" in output_text.lower().split() or "hello!" in output_text.lower().split() or "bye!" in output_text.lower().split() or "hey!" in output_text.lower().split():
                action = "handwave"
            elif "yes" in output_text.lower().split() or "correct" in output_text.lower().split() or "yea" in output_text.lower().split() or "yup" in output_text.lower().split() or "yep" in output_text.lower().split() or "yeah" in output_text.lower().split():
                action = "nod"
            elif "no" in output_text.lower().split() or "wrong" in output_text.lower().split():
                action = "headshake"
            elif "congratulations" in output_text.lower().split() or "great!" in output_text.lower().split():
                action = "clap"
            if action != "":
                print("action:", action)
            # --------------ACTION------------------------------------------------------------------------------

            # --------------EMOTION------------------------------------------------------------------------------
            if ":" in output_text:
                chatbot_text = output_text.split(":")[1].strip()
            else:
                chatbot_text = output_text
            # labels = ['joy', 'anger', 'fear', 'sadness', 'love', 'surprise']
            # labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
            model_id = "transformersbook/distilbert-base-uncased-finetuned-emotion"
            classifier = pipeline("text-classification", model=model_id)

            preds = classifier(chatbot_text, top_k=None)
            # preds = classifier(chatbot_text, return_all_scores=True)
            top_pred = classifier(chatbot_text, top_k=1)
            # print("input to emotion:", chatbot_text)
            print("chatbot emotion:", top_pred[0]["label"])

            preds_df = pd.DataFrame(preds)
            # preds_df = pd.DataFrame(preds[0])
            # plt.bar(labels, 100 * preds_df["score"], color='C0')
            plt.bar(preds_df["label"], 100 * preds_df["score"], color='C0')
            plt.title(f'"{chatbot_text}"')
            plt.ylabel("Class probability (%)")
            plt.show()
            # --------------EMOTION------------------------------------------------------------------------------

            # MAAB: Save user input history
            save_user_history(friend, cont)
            dialogue_history = './user_input_history_ai.txt'  # Create an empty list to store the dialogue history
            # Now you have the contents of the file stored in the dialogue_history list
            user_info = extract_user_info(dialogue_history, user_id)  # Extract the user info from the dialogue history
            write_user_info(user_info, "output.txt")  # Write the user info to a file
            converted_sentence = convert_user_info_to_sentences(user_info)  # Convert the user info to a sentence
            write_sentence_to_file(converted_sentence, "New_Prompt.txt")  # Write the sentence to a file
            count=0
            found = False
            for key in user_info:
                if key!= "age" and key!= "name" and key!= "user_id" and len(user_info[key])>0:
                    count +=1
            if count >=1 and user_info["age"] is not None:
                user_interests = []
                for key in user_info:
                    if key!= "age" and key!= "name" and key!= "user_id" and len(user_info[key])>0:
                        user_interests.extend(user_info[key])
                        # print(user_interests)
                if user_info["age"]<20:
                    for i in user_interests:
                        for new_friend in young_friends:
                            if i.lower() in friend_list[new_friend]:
                                friend = new_friend
                                found = True
                                break
                elif user_info["age"]<40:
                    for i in user_interests:
                        for new_friend in middle_friends:
                            if i.lower() in friend_list[new_friend]:
                                friend = new_friend
                                found = True
                                break
                else:
                    for i in user_interests:
                        for new_friend in old_friends:
                            if i.lower() in friend_list[new_friend]:
                                friend = new_friend
                                found = True
                                break

                if found == True:
                    with open('context_{}.txt'.format(friend.lower())) as f:
                        context_init = f.readlines()
                    dialogue_init = []
                    for i in context_init:
                        if "<s>" in i[:3]:
                            dialogue_init.append(i)
                    with open("dialogue_history.txt", "w") as f:
                        f.write(''.join(dialogue_init))

if __name__ == "__main__":
    main()

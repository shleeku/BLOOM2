import logging
import torch
import transformers
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from datasets import load_dataset
from flask import Flask, request, jsonify


def prompt_1(sentence: str) -> str:
    return f'''Given single chatbot responses, specify which of the action classes "Neutral", "Handwave", "Nod", "Head Shake", "Clap" best classifies the following responses.

Chatbot response: Sure, I can help with that. What kind of books do you like to read?
Best action classification: Neutral

Chatbot response: Goodbye then, have a lovely evening!
Best action classification: Handwave

Chatbot response: I see your point, and I agree. The dataset you mentioned should be suitable for your task.
Best action classification: Nod

Chatbot response: I'm afraid that's not quite right. You may want to review the information I provided earlier.
Best action classification: Head Shake

Chatbot response: Congratulations! That is a wonderful accomplishment!
Best action classification: Clap

Chatbot response: I'm here to assist you. What information are you looking for?
Best action classification: Neutral

Chatbot response: Hi there! Waving hello to you!
Best action classification: Handwave

Chatbot response: Yes, I understand your query and I'm processing the information now.
Best action classification: Nod

Chatbot response: Unfortunately, that's incorrect. The right answer is...
Best action classification: Head Shake

Chatbot response: Excellent work! Your progress is really impressive!
Best action classification: Clap

Chatbot response: {sentence}
Best action classification:'''


def generate_action(generator, sentence: str) -> str:
    generated_text = generator(prompt_1(sentence), max_new_tokens=1)[0]['generated_text']
    return generated_text.split()[-1]


app = Flask(__name__)


@app.route('/chat', methods=['POST'])
def chat():
    # get the message from the POST request body
    data = request.get_json()
    message = data.get('message')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    logger.info("Initializing model")
    torch.cuda.empty_cache()

    ### BLOOM ###
    data_dir = "./mldata2/cache/transformers/bloom/"
    # checkpoint = "bigscience/bloom-7b1"
    checkpoint = "bigscience/bloom-3b"
    model = BloomForCausalLM.from_pretrained(checkpoint, cache_dir=data_dir)
    tokenizer = BloomTokenizerFast.from_pretrained(checkpoint, cache_dir=data_dir)
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

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

            # --------------------------------------------------------------------------------------------
            # dialogue_text = tokenizer.decode(input_combined[0])
            # print(tokenizer.decode(input_combined[0]))
            # print(input_combined)
            output = tokenizer.decode(model.generate(input_combined,
                                                     max_length=result_length,
                                                     do_sample=True,
                                                     top_k=50,
                                                     top_p=0.8,
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

            # Generate action classification for the output text
            action_class = generate_action(generator, output_text)
            return jsonify({"action_class": action_class, "text": output_text})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
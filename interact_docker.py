import logging
import torch
import transformers
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from User_profile_detector import extract_user_info
from User_profile_detector import save_user_history
from transformers import pipeline
from matplotlib import pyplot as plt
import pandas as pd
from User_profile_detector import write_user_info
from User_profile_detector import convert_user_info_to_sentences
from User_profile_detector import write_sentence_to_file
from flask import Flask, request, jsonify, session
import threading
import os
import signal
import time

# Initialize flask
app = Flask(__name__)
app.secret_key = 'VIP701'

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

logger.info("Initializing model")

# Initialize CUDA
torch.cuda.empty_cache()
device = torch.device("cuda:3")

# Load language model
data_dir="./mldata2/cache/transformers/bloom/"
checkpoint = "bigscience/bloom-3b"
model = BloomForCausalLM.from_pretrained(checkpoint, cache_dir=data_dir).to(device)
tokenizer = BloomTokenizerFast.from_pretrained(checkpoint, cache_dir=data_dir)

# Initialize dialogue
with open('context_init.txt') as f:
    context_init = f.readlines()
dialogue_init = []
for i in context_init:
    if "<s>" in i[:3]:
        dialogue_init.append(i)
with open("dialogue_history.txt", "w") as f:
    f.write(''.join(dialogue_init))
with open("user_input_history_ai.txt", "w+") as f:
    f.write("")

# Define emotion classifier and model
model_id = "transformersbook/distilbert-base-uncased-finetuned-emotion"
classifier = pipeline("text-classification", model=model_id)

# Define 'friends'
names = ["Brad", "Jenny", "Jimmy", "John", "Laura", "Tyler", "Sky", "Lizzie", "Alice"]
friend_list = {
    "Brad": ["weightlifting", "drinking", "partying"],
    "Sky": ["volleyball", "skiing", "arts and crafts"],
    "Lizzie": ["movies", "reading", "going to the beach"],
    "Tyler": ["hanging out", "skateboarding", "surfing"],
    "Jenny": ["writing", "politics", "environmental issues"],
    "Jimmy": ["rock band", "singing", "philosophy"],
    "Laura": ["knitting", "skydiving", "classical music"],
    "Alice": ["ballet", "jazz music", "opera"],
    "John": ["western movies", "broadway musicals", "history"]
}
friend_group = {
    'young': ["Tyler", "Sky", "Lizzie"],
    'middle': ["Brad", "Jenny", "Jimmy"],
    'old': ["Laura", "John", "Alice"]
}

# Action class generator
def classify_action(output_text):
    action = "neutral"
    if "hi" in output_text.lower() or "hello" in output_text.lower() or "hey" in output_text.lower() or "bye" in output_text.lower():
        action = "handwave"
    elif "yes" in output_text.lower() or "correct" in output_text.lower():
        action = "nod"
    elif "no" in output_text.lower() or "wrong" in output_text.lower():
        action = "headshake"
    elif "congratulations" in output_text.lower() or "great!" in output_text.lower():
        action = "clap"
    return action

def generate_response(tokenizer, model, device, prompt_context, prompt_dialogue, converted_sentence, result_length=2048):
    # Tokenize the prompts
    tokens_context = tokenizer.tokenize(prompt_context)
    tokens_dialogue = tokenizer.tokenize(prompt_dialogue)
    # Convert tokens to ids
    ids_context = tokenizer.convert_tokens_to_ids(tokens_context)
    ids_dialogue = tokenizer.convert_tokens_to_ids(tokens_dialogue)
    # Ensure dialogue length does not exceed maximum length
    max_len_dialogue = result_length - len(ids_context)
    if len(ids_dialogue) > max_len_dialogue:
        ids_dialogue = ids_dialogue[-max_len_dialogue:]
    # Combine the context and dialogue
    ids_combined = ids_context + ids_dialogue
    # Add additional sentences if available
    if converted_sentence:
        new_sentences = converted_sentence[-1]
        new_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(new_sentences))
        ids_combined += new_ids

    # Convert ids to tensor format
    input_combined = torch.tensor([ids_combined])
    # Initialize variables
    found = False
    counter = 0
    output = "I don't know what to say."
    # Sampling loop
    while not found and counter < 25:
        counter += 1
        total_output = model.generate(input_combined.to(device),
                                      max_length=result_length,
                                      do_sample=True,
                                      top_k=100,
                                      top_p=0.70,
                                      num_return_sequences=1,
                                      temperature=0.8)
        output_sequences = [tokenizer.decode(sequence, skip_special_tokens=True) for sequence in total_output]
        for i in output_sequences:
            sequence_list = list(i.split("\n"))
            if sequence_list[-1].strip() != "" and "user:" not in sequence_list[-1].lower() and sequence_list[-1].strip() != "AI:":
                output = i
                found = True
                break
    return output

def assign_friend(user_info, user_interests, friend_list, friend_group):
    age_group = ''
    if user_info["age"] < 20:
        age_group = 'young'
    elif user_info["age"] < 40:
        age_group = 'middle'
    else:
        age_group = 'old'
    
    found = False
    friend = None
    for interest in user_interests:
        for new_friend in friend_group[age_group]:
            if interest.lower() in friend_list[new_friend]:
                friend = new_friend
                found = True
                break
        if found:
            break

    return friend, found

def update_dialogue_history(cont, output):
    output_list = list(output.split("\n"))
    with open("dialogue_history.txt", "a") as f:
        f.write(cont + output_list[-1] + "\n")
    return output_list

def parse_output(output_list):
    if output_list[-1][:3] == "<s>":
        output_text = output_list[-1][3:]
    else:
        output_text = output_list[-1]
    if output_list[-1][-4:] == "</s>":
        output_text = output_text[:-4]

    if ":" in output_text:
        output_text = output_text.split(":")[1].strip()

    return output_text

def extract_and_save_user_info(dialogue_history, user_id):
    user_info = extract_user_info(dialogue_history, user_id)  # Extract the user info from the dialogue history
    write_user_info(user_info, "output.txt")  # Write the user info to a file
    converted_sentence = convert_user_info_to_sentences(user_info)  # Convert the user info to a sentence
    write_sentence_to_file(converted_sentence, "New_Prompt.txt")  # Write the sentence to a file
    return user_info

# Define shutdown sequence
def shutdown_server():
    """Shut down the server from a different thread"""
    time.sleep(5)
    os.kill(os.getpid(), signal.SIGINT)

def async_shutdown_server():
    """Create a new thread to shut down the server"""
    threading.Thread(target=shutdown_server).start()

@app.route('/chat', methods=['POST'])
def chat():
    # get the message from the POST request body
    data = request.get_json()

    id_received = False
    while not id_received:
        try:
            user_id = int(data.get("User_ID"))
            id_received = True
        except ValueError:
            print("Integer User ID required.")
    message = data.get('message')

    # Session management
    if 'chat_history' not in session:
        session['chat_history'] = ''
    session['chat_history'] += ' ' + message  

    friend = "AI"
    response_complete = False
    flag = True
    while flag:
        breakout = False
        cont = "<s> User: " + message + " </s>\n"
    # Identify friend persona
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
                output_text = "{}: Hey it's {}!".format(friend, friend)
                response_complete = True
        if breakout == True:
            continue

    # Initialize dialogue
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
        prompt_context = ''.join(context)
        prompt_dialogue = ''.join(dialogue)

        # Check end of conversation
        if "bye" in cont:
            output_text = 'good bye'
            response_complete = True 
            async_shutdown_server()
        if response_complete == True:
            flag = False
        else:
            # Generate response from language model
            converted_sentence = []
            output = generate_response(tokenizer, model, device, prompt_context, prompt_dialogue, converted_sentence)
            output_list = update_dialogue_history(cont, output)
            output_text = parse_output(output_list)

            # Save and access user data
            save_user_history(friend, cont)
            dialogue_history = './user_input_history_ai.txt'  
            user_info = extract_and_save_user_info(dialogue_history, user_id)

            # Persona matching algorithm
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
                    friend, found = assign_friend(user_info, user_interests, friend_list, friend_group)
                    print(friend)

                if found == True:
                    with open('context_{}.txt'.format(friend.lower())) as f:
                        context_init = f.readlines()
                    dialogue_init = []
                    for i in context_init:
                        if "<s>" in i[:3]:
                            dialogue_init.append(i)
                    with open("dialogue_history.txt", "w") as f:
                        f.write(''.join(dialogue_init))
            
        # Action class generation
        action = classify_action(output_text)

        # Emotion classifier
        preds = classifier(output_text, top_k=None)
        top_pred = classifier(output_text, top_k=1)
        emotion = top_pred[0]["label"]

        return jsonify({"text": output_text, "action": action, "emotion": emotion} )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
import spacy
import re
import json
from dateutil.parser import parse
from datetime import date
import nltk
from nltk.corpus import names



#nltk.download('names')
#nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

def save_user_history(friend, user_input):
    file_path = "user_input_history_ai.txt".format(friend.lower())

    # Write the new user input to the file
    with open(file_path, "a") as f:
        f.write(user_input + "\n")


def extract_user_info(file_path, user_id):

    user_info = {
            'user_id': user_id,
            'name': None,
            'age': None,
            'jobs': [],
            'hobbies': [],
            'interests': [],
            'favorite_sports': [],
        }

    with open(file_path, 'r') as file:
        lines = file.readlines()


        # Remove <s> tags from each line
        lines = [re.sub(r'<s>', '', line) for line in lines]
        lines = [re.sub(r'</s>', '', line) for line in lines]
        user_input = ' '.join(lines)  # Join lines into a single string
        doc = nlp(user_input.lower())

    # Define a set of hobby keywords
    hobby_keywords ={'puzzles', 'crossword puzzles', 'sudoku', 'jigsaw puzzles', 'model building', 'fishing', 'hiking', 'camping', 'biking', 'running', 'swimming', 'surfing', 'skateboarding', 'snowboarding', 'skiing', 'rock climbing', 'yoga', 'pilates', 'meditation', 'tai chi', 'golf', 'tennis', 'badminton', 'table tennis', 'football', 'basketball', 'baseball', 'soccer', 'volleyball', 'cricket', 'rugby', 'hockey', 'gymnastics', 'boxing', 'martial arts', 'archery', 'target shooting', 'fencing', 'horseback riding', 'diving', 'waterskiing', 'wakeboarding', 'kayaking', 'canoeing', 'sailing', 'paintball', 'airsoft', 'kite flying', 'birdwatching', 'star gazing', 'astrophotography', 'magic tricks', 'coin collecting', 'stamp collecting', 'scuba diving', 'snorkeling', 'surf fishing', 'skydiving', 'paragliding', 'bungee jumping', 'zip lining', 'amateur radio', 'hiking', 'cycling', 'mountain biking', 'motorcycling', 'off-roading', 'car racing', 'cooking', 'baking', 'mixology', 'wine tasting', 'beer brewing', 'coffee roasting', 'tea tasting', 'candle making', 'soap making', 'pottery', 'sculpting', 'jewelry making', 'leatherworking', 'wood carving', 'metalworking', 'glassblowing', 'embroidery', 'cross-stitching', 'weaving', 'macrame', 'quilting', 'knitting', 'crocheting', 'sewing', 'fashion design', 'cosplay', 'drawing', 'painting', 'photography', 'graphic design', 'digital art', 'writing', 'poetry', 'journaling', 'creative writing', 'blogging', 'storytelling', 'reading', 'book club', 'bookbinding', 'calligraphy', 'music', 'playing an instrument', 'singing', 'songwriting', 'music production', 'djing', 'dancing', 'ballet', 'hip hop', 'salsa', 'ballroom', 'contemporary', 'street dance', 'zumba', 'pilates', 'yoga', 'meditation', 'tai chi', 'aerobics', 'piloxing', 'zumba', 'cardio kickboxing', 'boxing', 'martial arts', 'fitness training', 'weightlifting', 'parkour', 'circus arts', 'acrobatics', 'trampoline', 'fire spinning', 'hula hooping', 'bowling', 'billiards', 'darts', 'table soccer', 'laser tag', 'escape rooms', 'astrology', 'tarot reading', 'chess', 'backgammon', 'poker', 'bridge', 'mahjong', 'origami', 'juggling', 'cycling', 'swimming', 'running', 'hiking', 'rowing', 'tennis', 'basketball', 'football', 'volleyball', 'cricket', 'golf', 'reading', 'writing', 'painting', 'drawing', 'photography', 'cooking', 'baking', 'gardening', 'singing', 'singer', 'sing', 'rock band', 'guitarist', 'guitar', 'bassist', 'bass', 'drummer', 'drumming', 'drums', 'vocals', 'vocalist', 'reading', 'writing', 'arts and crafts', }

    # Define a set of job keywords
    sports_keywords = {'surf life saving', 'barefoot skiing', 'synchronized skating', 'tug of war', 'sepak takraw', 'kabaddi', 'ski jumping', 'luge', 'biathlon', 'speed skating', 'hapkido', 'street hockey', 'cycle polo', 'paddleboarding', 'korfball', 'jai alai', 'judo', 'bandy', 'racquetball', 'squash', 'trampolining', 'bowls', 'trampoline gymnastics', 'floorball', 'underwater hockey', 'boccia', 'fistball', 'ski orienteering', 'powerboating', 'dog agility', 'lawn mower racing', 'footgolf', 'mountain unicycling', 'bossaball', 'cycle speedway', 'cycle polo', 'kickball', 'disk golf', 'quidditch', 'camogie', 'shinty', 'kabaddi', 'kho kho', 'tent pegging', 'beach handball', 'futsal', 'korfball', 'kin-ball', 'padel', 'ringette', 'roller derby', 'sepak takraw', 'ultimate frisbee', 'gaelic football', 'hurling', 'fives', 'canoe polo', 'powerchair football', 'wheelchair basketball', 'goalball', 'wheelchair rugby', 'sitting volleyball', 'blind cricket', 'sledge hockey', 'sitting volleyball', 'ski mountaineering', 'boardercross', 'skeleton', 'bobsleigh', 'floorball', 'ice climbing', 'artistic roller skating', 'rhythmic gymnastics', 'wheelchair tennis', 'tchoukball', 'paralympic athletics', 'paralympic swimming', 'paralympic cycling', 'paralympic archery', 'paralympic powerlifting', 'paralympic goalball', 'paralympic judo', 'paralympic rowing', 'paralympic equestrian', 'paralympic sailing', 'paralympic volleyball', 'paralympic wheelchair basketball', 'paralympic wheelchair rugby', 'paralympic wheelchair fencing', 'paralympic wheelchair tennis', 'paralympic table tennis', 'paralympic wheelchair curling', 'paralympic wheelchair dance sport', 'paralympic sitting volleyball', 'paralympic biathlon', 'paralympic alpine skiing', 'paralympic nordic skiing', 'paralympic snowboarding', 'paralympic sledge hockey', 'paralympic wheelchair rugby league', 'paralympic equestrian vaulting', 'paralympic wheelchair archery', 'paralympic sitting volleyball', 'paralympic boccia', 'paralympic rugby union''football', 'basketball', 'baseball', 'soccer', 'tennis', 'golf', 'cricket', 'rugby', 'volleyball', 'hockey', 'table tennis', 'badminton', 'swimming', 'swim', 'athletics', 'boxing', 'martial arts', 'wrestling', 'cycling', 'gymnastics', 'ice hockey', 'handball', 'snooker', 'darts', 'squash', 'water polo', 'rowing', 'skiing', 'snowboarding', 'surfing', 'skateboarding', 'netball'}

    # Define a list of interests
    interests_keywords = {'podcasts', 'science outreach', 'ai', 'nlp', 'computer vision', 'cv', 'gaming', 'cooking', 'science education', 'creative coding', 'internet culture', 'digital marketing', 'social media marketing', 'content creation', 'graphic novels', 'comic art', 'street art', 'fashion styling', 'vintage collecting', 'documentary films', 'historical research', 'urban exploration', 'adventure travel', 'wildlife photography', 'cultural festivals', 'food photography', 'plant-based cooking', 'home brewing', 'sustainable fashion', 'outdoor survival skills', 'car restoration', 'extreme sports', 'off-roading', 'film photography', 'cocktail mixology', 'woodworking', 'diy electronics', 'mobile photography', 'space exploration', 'stargazing', 'meteorology', 'fitness coaching', 'powerlifting', 'bodybuilding', 'yoga instruction', 'holistic healing', 'herbal medicine', 'tarot card reading', 'witchcraft', 'paranormal investigations', 'stand-up comedy', 'improvisational theater', 'political activism', 'human rights advocacy', 'gender studies', 'intersectional feminism', 'community gardening', 'permaculture', 'renewable energy', 'sustainable architecture', 'dance choreography', 'contemporary art', 'street photography', 'virtual reality gaming', 'app development', 'ethical fashion', 'social entrepreneurship', 'green living', 'zero waste lifestyle', 'minimalism', 'concert photography', 'musical theatre', 'voice acting', 'screenwriting', 'historical fiction', 'astrophotography', 'cryptography', 'ethical hacking', 'podcasting', 'language learning', 'calligraphy', 'typography', 'sketching', 'comic book writing', 'freelance writing', 'nature conservation', 'marine biology', 'birdwatching', 'animal rescue', 'astronomy', 'archaeology', 'geology', 'geography', 'environmental science', 'human anatomy', 'classical music', 'opera', 'street dancing', 'salsa dancing', 'fashion illustration', 'vintage fashion', 'culinary tourism', 'plant care', 'sustainable gardening', 'adventure sports', 'backpacking', 'mountaineering', 'parasailing', 'kayaking', 'sailing', 'volunteer work', 'youth mentoring', 'nonprofit organizations', 'public health', 'nutrition', 'life coaching', 'positive psychology', 'personal development', 'mindset coaching', 'meditation retreats', 'spiritual healing', 'holistic nutrition', 'eastern philosophy', 'ancient history', 'world mythology', 'fine art photography', 'documentary photography', 'sustainable travel', 'cultural anthropology', 'linguistics', 'language translation', 'robotics engineering', 'artificial intelligence', 'quantum physics', 'data analytics', 'virtual assistant development', 'mobile app design', 'game design', 'bioengineering', 'space engineering', 'renewable energy engineering', 'cognitive psychology', 'neuropsychology', 'philosophy of mind', 'environmental ethics', 'neuroethics', 'bioethics', 'social psychology', 'political philosophy', 'ethics in technology', 'science fiction writing', 'fantasy literature', 'cultural studies', 'art history', 'cosmology', 'quantum computing', 'behavioral economics', 'market research', 'social work', 'community development', 'gender equality', 'inclusive design', 'environmental activism', 'sustainable agriculture', 'climate change mitigation','ai', 'philosophy', 'politics', 'movies', 'beach', 'hanging out', 'environmental', 'environmental issues', 'environment', 'drinking', 'partying', 'classical', 'jazz', 'Western movies', 'Broadway'}


    # Define a set of job keywords
    job_keywords = {'doctor','teacher','accountant', 'administrative assistant', 'software engineer', 'marketing manager', 'graphic designer', 'customer service representative', 'project manager', 'sales associate', 'human resources specialist', 'financial analyst', 'data scientist', 'business development manager', 'operations coordinator', 'public relations coordinator', 'event planner', 'content writer', 'operations analyst', 'quality assurance engineer', 'social media specialist', 'product manager', 'it support specialist', 'legal assistant', 'research analyst', 'logistics coordinator', 'digital marketing manager', 'hr generalist', 'financial controller', 'technical writer', 'supply chain analyst', 'operations manager', 'market research analyst', 'ux/ui designer', 'account manager', 'sales executive', 'data entry clerk', 'marketing coordinator', 'medical assistant', 'web designer', 'dog trainer', 'president', 'software developer', 'data analyst', 'financial advisor', 'customer support specialist', 'operations manager', 'business analyst', 'registered nurse', 'event coordinator', 'digital marketing specialist', 'it project manager', 'sales representative', 'content manager', 'hr manager', 'operations supervisor', 'marketing specialist', 'executive assistant', 'front end developer', 'financial planner', 'customer success manager', 'supply chain manager', 'quality control inspector', 'social media manager', 'product marketing manager', 'accounting clerk', 'administrative coordinator', 'database administrator', 'market research manager', 'ux designer', 'financial analyst', 'customer service manager', 'business development coordinator', 'operations analyst', 'public relations manager', 'event manager', 'content strategist', 'data scientist', 'it consultant', 'legal secretary', 'research assistant', 'logistics manager', 'digital marketing coordinator', 'hr assistant', 'finance manager', 'technical support specialist', 'supply chain coordinator', 'operations director', 'market analyst', 'ui designer', 'accounting manager', 'sales manager', 'data analyst', 'marketing analyst', 'medical receptionist', 'web developer', 'dog groomer', 'ceo', 'software architect', 'business intelligence analyst', 'financial controller', 'customer service supervisor', 'project coordinator', 'sales coordinator', 'content writer', 'hr coordinator', 'operations assistant', 'marketing assistant', 'executive secretary', 'back end developer', 'investment analyst', 'customer success specialist', 'supply chain specialist', 'quality assurance analyst', 'social media coordinator', 'product owner', 'accounting assistant', 'administrative secretary', 'database developer', 'market research analyst', 'ux researcher', 'financial advisor', 'customer support representative', 'operations supervisor', 'business analyst', 'registered nurse', 'event planner', 'digital marketing specialist', 'it project coordinator', 'sales associate', 'content marketing manager', 'hr director', 'operations manager', 'marketing manager', 'executive director', 'finance analyst', 'technical writer', 'supply chain manager', 'quality control technician', 'social media specialist', 'product manager', 'account executive', 'sales representative', 'data entry specialist', 'marketing coordinator', 'medical secretary', 'web designer', 'dog walker', 'chief operating officer', 'software engineer', 'data scientist', 'financial planner', 'customer support manager', 'supply chain analyst', 'operations analyst', 'public relations specialist', 'event coordinator', 'content strategist', 'data analyst', 'it manager', 'legal assistant', 'research analyst', 'logistics coordinator', 'digital marketing manager', 'hr manager', 'financial controller', 'technical writer', 'supply chain analyst', 'operations manager', 'market research analyst', 'ux/ui designer', 'developer', 'engineer', 'designer', 'manager', 'analyst', 'student'}

    names_patterns = [
        r"My name is ([A-Za-z]+)",
        r"I'm ([A-Za-z]+)",
        r"Allow me to introduce myself, I'm ([A-Za-z]+)",
        r"Hi, I go by the name of ([A-Za-z]+)",
        r"Pleased to meet you, I'm ([A-Za-z]+)",
        r"They call me ([A-Za-z]+)",
        r"You can call me ([A-Za-z]+)",
        r"Let me introduce myself, I am ([A-Za-z]+)",
        r"The name's ([A-Za-z]+)",
        r"I go by ([A-Za-z]+)"
    ]

    # Check if any name pattern matches in the input text
    for pattern in names_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            matched_string = nlp(match.group(1))
            if len(matched_string.ents) > 0 and matched_string.ents[0].label_ == "PERSON":
                user_name =matched_string.ents[0].text.capitalize()
                user_info['name'] = user_name.strip()
            else:
                tokenized_text = nltk.word_tokenize(match.group(1))
                for word in tokenized_text:
                    if word in names.words():
                        user_name = word.capitalize()
                        user_info['name'] = user_name.strip()

    for token in doc:

        # Check if the token text is in the set of hobby keywords
        if token.text.lower() in hobby_keywords:
            # Add the token as a hobby
            user_info['hobbies'].append(token.text)

        # Check if the token is in the list of interests
        if token.text in interests_keywords:
            # Add the token as an interest
            user_info['interests'].append(token.text)  # Add the token to the user interests



        # Check if the token text is in the set of sports keywords
        if token.text.lower() in sports_keywords:
            # Add the token as a sport
            user_info['favorite_sports'].append(token.text)


        # Check if the token is a number
        if token.like_num:
            # Check the dependency tag of the token
            if token.dep_ == 'nummod':
                # Check if the head token is 'years'
                if token.head.text == 'years':
                    age = int(token.text)
                    user_info['age'] = age
        # Define the regular expression pattern
        pattern = r"[Ii]'?m (\d+)"

        # Use re.search to find the match
        match = re.search(pattern, user_input, re.IGNORECASE)  # Case-insensitive search

        # Check if there's a match and extract the age
        if match:
            age = int(match.group(1))
            user_info['age'] = age





        # Check if the token text is in the set of job keywords
        if token.text.lower() in job_keywords:
            # Add the token as a job
             user_info['jobs'].append(token.text)



    return user_info
import json

def write_user_info(user_info, output_file_path):
    try:
        with open(output_file_path, 'r+') as file:
            lines = file.readlines()
            file.seek(0)
            for line in lines:
                line = line.strip()
                if line:
                    stored_user_info = json.loads(line)
                    if stored_user_info["user_id"] != user_info["user_id"]:
                        file.write(line + '\n')
            file.truncate()
            user_info_str = json.dumps(user_info)
            file.write(user_info_str + '\n')
    except (IOError, json.JSONDecodeError) as e:
        print(f"An error occurred while writing user info: {str(e)}")


def convert_user_info_to_sentences(user_info):
    sentences = set()

    if "name" in user_info and user_info["name"]:
        sentences.add("The user name is {}.".format(user_info["name"]))
    if "user_id" in user_info and user_info["user_id"] is not None:
        sentences.add("{}'s id is {}.".format(user_info["name"],user_info["user_id"]))
    if "age" in user_info and user_info["age"] is not None:
        sentences.add(" {}'s age is {}.".format(user_info["name"],user_info["age"]))
    if "jobs" in user_info and user_info["jobs"]:
        sentences.add("{}'s job is {}.".format(user_info["name"],", ".join(user_info["jobs"])))
    if "hobbies" in user_info and user_info["hobbies"]:
        sentences.add("{}'s hobbies are {}.".format(user_info["name"],", ".join(user_info["hobbies"])))
    if "interests" in user_info and user_info["interests"]:
        sentences.add("{} is interested in {}.".format(user_info["name"],", ".join(user_info["interests"])))
    if "favorite_sports" in user_info and user_info["favorite_sports"]:
        sentences.add("{} favorite sports are {}.".format(user_info["name"],", ".join(user_info["favorite_sports"])))

    return list(sentences)



def write_sentence_to_file(sentences, output_file_path):
    with open(output_file_path, 'r') as file:
        existing_sentences = set(file.read().splitlines())

    with open(output_file_path, 'a') as file:
        for sentence in sentences:
            if sentence not in existing_sentences:
                file.write(sentence + '\n')
                existing_sentences.add(sentence)
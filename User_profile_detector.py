import spacy
import re
import json
from dateutil.parser import parse
from datetime import date
import nltk
from nltk.corpus import names
nltk.download('names')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

def save_user_history(friend, user_input):
    file_path = "user_input_history_{}.txt".format(friend.lower())

    # Write the new user input to the file
    with open(file_path, "a") as f:
        f.write(user_input + "\n")


def extract_user_info(file_path):

    user_info = {
            'name': None,
            'age': None,
            'jobs': [],
            'hobbies': [],
            'interests': [],
            'favorite_sports': [],
        }

    with open(file_path, 'r') as file:
        lines = file.readlines()
        user_input = ' '.join(lines)  # Join lines into a single string
        doc = nlp(user_input.lower())

        # Remove <s> tags from each line
        lines = [re.sub(r'<s>', '', line) for line in lines]
        lines = [re.sub(r'</s>', '', line) for line in lines]

    # Define a set of hobby keywords
    hobby_keywords = {'Puzzles', 'Crossword Puzzles', 'Sudoku', 'Jigsaw Puzzles', 'Model Building', 'Fishing', 'Hiking', 'Camping', 'Biking', 'Running', 'Swimming', 'Surfing', 'Skateboarding', 'Snowboarding', 'Skiing', 'Rock Climbing', 'Yoga', 'Pilates', 'Meditation', 'Tai Chi', 'Golf', 'Tennis', 'Badminton', 'Table Tennis', 'Football', 'Basketball', 'Baseball', 'Soccer', 'Volleyball', 'Cricket', 'Rugby', 'Hockey', 'Gymnastics', 'Boxing', 'Martial Arts', 'Archery', 'Target Shooting', 'Fencing', 'Horseback Riding', 'Diving', 'Waterskiing', 'Wakeboarding', 'Kayaking', 'Canoeing', 'Sailing', 'Paintball', 'Airsoft', 'Kite Flying', 'Birdwatching', 'Star Gazing', 'Astrophotography', 'Magic Tricks', 'Coin Collecting', 'Stamp Collecting', 'Scuba Diving', 'Snorkeling', 'Surf Fishing', 'Skydiving', 'Paragliding', 'Bungee Jumping', 'Zip Lining', 'Amateur Radio', 'Hiking', 'Cycling', 'Mountain Biking', 'Motorcycling', 'Off-Roading', 'Car Racing', 'Cooking', 'Baking', 'Mixology', 'Wine Tasting', 'Beer Brewing', 'Coffee Roasting', 'Tea Tasting', 'Candle Making', 'Soap Making', 'Pottery', 'Sculpting', 'Jewelry Making', 'Leatherworking', 'Wood Carving', 'Metalworking', 'Glassblowing', 'Embroidery', 'Cross-Stitching', 'Weaving', 'Macrame', 'Quilting', 'Knitting', 'Crocheting', 'Sewing', 'Fashion Design', 'Cosplay', 'Drawing', 'Painting', 'Photography', 'Graphic Design', 'Digital Art', 'Writing', 'Poetry', 'Journaling', 'Creative Writing', 'Blogging', 'Storytelling', 'Reading', 'Book Club', 'Bookbinding', 'Calligraphy', 'Music', 'Playing an Instrument', 'Singing', 'Songwriting', 'Music Production', 'DJing', 'Dancing', 'Ballet', 'Hip Hop', 'Salsa', 'Ballroom', 'Contemporary', 'Street Dance', 'Zumba', 'Pilates', 'Yoga', 'Meditation', 'Tai Chi', 'Aerobics', 'Piloxing', 'Zumba', 'Cardio Kickboxing', 'Boxing', 'Martial Arts', 'Fitness Training', 'Weightlifting', 'Parkour', 'Circus Arts', 'Acrobatics', 'Trampoline', 'Fire Spinning', 'Hula Hooping', 'Bowling', 'Billiards', 'Darts', 'Table Soccer', 'Laser Tag', 'Escape Rooms', 'Astrology', 'Tarot Reading', 'Chess', 'Backgammon', 'Poker', 'Bridge', 'Mahjong', 'Origami', 'Juggling', 'Cycling', 'Swimming', 'Running', 'Hiking', 'Rowing', 'Tennis', 'Basketball', 'Football', 'Volleyball', 'Cricket', 'Golf''Reading', 'Writing', 'Painting', 'Drawing', 'Photography', 'Cooking', 'Baking', 'Gardening', 'Playing an Instrument', 'Singing', 'Dancing', 'Acting', 'Listening to Music', 'Watching Movies', 'Watching TV Series', 'Playing Video Games', 'Collecting Stamps', 'Collecting Coins', 'Collecting Antiques', 'Collecting Comic Books', 'Collecting Sports Memorabilia', 'Woodworking', 'Knitting', 'Crocheting', 'Sewing', 'Embroidery', 'Quilting', 'Pottery', 'Scrapbooking', 'Calligraphy', 'Origami', 'Chess', 'Playing Card Games', 'Board Games', 'Puzzles', 'Crossword Puzzles', 'Sudoku', 'Jigsaw Puzzles', 'Model Building', 'Fishing', 'Hiking', 'Camping', 'Biking', 'Running', 'Swimming', 'Surfing', 'Skateboarding', 'Snowboarding', 'Skiing', 'Rock Climbing', 'Yoga', 'Pilates', 'Meditation', 'Tai Chi', 'Golf', 'Tennis', 'Badminton', 'Table Tennis', 'Football', 'Basketball', 'Baseball', 'Soccer', 'Volleyball', 'Cricket', 'Rugby', 'Hockey', 'Gymnastics', 'Boxing', 'Martial Arts', 'Archery', 'Target Shooting', 'Fencing', 'Horseback Riding', 'Diving', 'Waterskiing', 'Wakeboarding', 'Kayaking', 'Canoeing', 'Sailing', 'Paintball', 'Airsoft', 'Kite Flying', 'Birdwatching', 'Star Gazing', 'Astrophotography', 'Magic Tricks', 'Coin Collecting', 'Stamp Collecting', 'Scuba Diving', 'Snorkeling', 'Surf Fishing', 'Skydiving', 'Paragliding', 'Bungee Jumping', 'Zip Lining', 'Amateur Radio', 'Hiking', 'Cycling', 'Mountain Biking', 'Motorcycling', 'Off-Roading', 'Car Racing', 'Cooking', 'Baking', 'Mixology', 'Wine Tasting', 'Beer Brewing', 'Coffee Roasting', 'Tea Tasting', 'Candle Making', 'Soap Making', 'Pottery', 'Sculpting', 'Jewelry Making', 'Leatherworking', 'Wood Carving', 'Metalworking', 'Glassblowing', 'Embroidery', 'Cross-Stitching', 'Weaving', 'Macrame', 'Quilting', 'Knitting', 'Crocheting', 'Sewing', 'Fashion Design', 'Cosplay', 'Drawing', 'Painting', 'Photography', 'Graphic Design', 'Digital Art', 'Writing', 'Poetry', 'Journaling', 'Creative Writing', 'Blogging', 'Storytelling', 'Reading', 'Book Club', 'Bookbinding', 'Calligraphy', 'Music', 'Playing an Instrument', 'Singing', 'Songwriting', 'Music Production', 'DJing', 'Dancing', 'Ballet', 'Hip Hop', 'Salsa', 'Ballroom', 'Contemporary', 'Street Dance', 'Zumba', 'Pilates', 'Yoga', 'Meditation', 'Tai Chi', 'Aerobics', 'Piloxing', 'Zumba', 'Cardio Kickboxing', 'Boxing', 'Martial Arts''playing', 'painting', 'reading', 'soccer', 'swimming'}

    # Define a set of sports keywords
    sports_keywords = {'Surf Life Saving', 'Barefoot Skiing', 'Synchronized Skating', 'Tug of War', 'Sepak Takraw', 'Kabaddi', 'Ski Jumping', 'Luge', 'Biathlon', 'Speed Skating', 'Hapkido', 'Street Hockey', 'Cycle Polo', 'Paddleboarding', 'Korfball', 'Jai Alai', 'Judo', 'Bandy', 'Racquetball', 'Squash', 'Trampolining', 'Bowls', 'Trampoline Gymnastics', 'Floorball', 'Underwater Hockey', 'Boccia', 'Fistball', 'Ski Orienteering', 'Powerboating', 'Dog Agility', 'Lawn Mower Racing', 'Footgolf', 'Mountain Unicycling', 'Bossaball', 'Cycle Speedway', 'Cycle Polo', 'Kickball', 'Disk Golf', 'Quidditch', 'Camogie', 'Shinty', 'Kabaddi', 'Kho Kho', 'Tent Pegging', 'Beach Handball', 'Futsal', 'Korfball', 'Kin-Ball', 'Padel', 'Ringette', 'Roller Derby', 'Sepak Takraw', 'Ultimate Frisbee', 'Gaelic Football', 'Hurling', 'Fives', 'Canoe Polo', 'Powerchair Football', 'Wheelchair Basketball', 'Goalball', 'Wheelchair Rugby', 'Sitting Volleyball', 'Blind Cricket', 'Sledge Hockey', 'Sitting Volleyball', 'Ski Mountaineering', 'Boardercross', 'Skeleton', 'Bobsleigh', 'Floorball', 'Ice Climbing', 'Artistic Roller Skating', 'Rhythmic Gymnastics', 'Wheelchair Tennis', 'Tchoukball', 'Paralympic Athletics', 'Paralympic Swimming', 'Paralympic Cycling', 'Paralympic Archery', 'Paralympic Powerlifting', 'Paralympic Goalball', 'Paralympic Judo', 'Paralympic Rowing', 'Paralympic Equestrian', 'Paralympic Sailing', 'Paralympic Volleyball', 'Paralympic Wheelchair Basketball', 'Paralympic Wheelchair Rugby', 'Paralympic Wheelchair Fencing', 'Paralympic Wheelchair Tennis', 'Paralympic Table Tennis', 'Paralympic Wheelchair Curling', 'Paralympic Wheelchair Dance Sport', 'Paralympic Sitting Volleyball', 'Paralympic Biathlon', 'Paralympic Alpine Skiing', 'Paralympic Nordic Skiing', 'Paralympic Snowboarding', 'Paralympic Sledge Hockey', 'Paralympic Wheelchair Rugby League', 'Paralympic Equestrian Vaulting', 'Paralympic Wheelchair Archery', 'Paralympic Sitting Volleyball', 'Paralympic Boccia', 'Paralympic Rugby Union''Football', 'Basketball', 'Baseball', 'Soccer', 'Tennis', 'Golf', 'Cricket', 'Rugby', 'Volleyball', 'Hockey', 'Table Tennis', 'Badminton', 'Swimming', 'Athletics', 'Boxing', 'Martial Arts', 'Wrestling', 'Cycling', 'Gymnastics', 'Ice Hockey', 'Handball', 'Snooker', 'Darts', 'Squash', 'Water Polo', 'Rowing', 'Skiing', 'Snowboarding', 'Surfing', 'Skateboarding', 'Netball', 'Softball', 'Bowling', 'Fencing', 'Archery', 'Equestrian', 'Polo', 'Racquetball', 'Triathlon', 'Karate', 'Taekwondo', 'Judo', 'Canoeing', 'Kayaking', 'Weightlifting', 'Powerlifting', 'Ultimate Frisbee', 'Rugby Sevens', 'Beach Volleyball', 'Paddleboarding', 'Cricket', 'Gaelic Football', 'Hurling', 'Lacrosse', 'Bobsleigh', 'Skeleton', 'Bungee Jumping', 'Climbing', 'Hang Gliding', 'Paragliding', 'Kiteboarding', 'Wingsuit Flying', 'Skydiving', 'Sailing', 'Windsurfing', 'Kite Surfing', 'Wakeboarding', 'Water Skiing', 'Jet Skiing', 'Motor Racing', 'Formula 1', 'NASCAR', 'IndyCar', 'Motorcycle Racing', 'Motocross', 'BMX Racing', 'Rallying', 'Mountain Biking', 'Street Skateboarding', 'Vert Skateboarding', 'Figure Skating', 'Speed Skating', 'Short Track Speed Skating', 'Inline Skating', 'Roller Skating', 'Ice Dancing', 'Ice Cross Downhill', 'Barefoot Waterskiing', 'Synchronized Swimming', 'Diving', 'High Jump', 'Long Jump', 'Triple Jump', 'Pole Vault', 'Shot Put', 'Discus Throw', 'Hammer Throw', 'Javelin Throw', 'Decathlon', 'Heptathlon', 'Marathon', 'Sprint', 'Hurdles', 'Relay Race', 'Cross Country Running', 'Race Walking', 'Mountain Climbing', 'Trail Running', 'Orienteering', 'Parkour', 'Yoga', 'Pilates', 'Zumba', 'Aerobics', 'Step Aerobics', 'Kickboxing', 'Spinning', 'Bodybuilding', 'Powerlifting', 'CrossFit', 'Paddle Tennis', 'Platform Tennis', 'Racquetball', 'Squash', 'Beach Tennis', 'Frisbee Golf', 'Pickleball', 'Croquet', 'Bocce Ball', 'Shuffleboard', 'Lawn Bowls', 'Horse Racing', 'Horseback Riding', 'Polo', 'Synchronized Riding', 'Dressage', 'Show Jumping', 'Vaulting', 'Car Racing', 'Go-Kart Racing', 'Radio-Controlled Racing', 'Slot Car Racing', 'Gaming', 'Esports', 'Billiards', 'Pool', 'Snooker', 'Darts', 'Foosball', 'Chess', 'Checkers', 'Table Hockey', 'Bowling', 'Air Hockey', 'Curling', 'Petanque', 'Fishing', 'Angling', 'Fly Fishing', 'soccer', 'basketball', 'football', 'climbing', 'tennis', 'hiking'}

    # Define a list of interests
    interests_keywords = ['Science Education', 'Creative Coding', 'Internet Culture', 'Digital Marketing', 'Social Media Marketing', 'Content Creation', 'Graphic Novels', 'Comic Art', 'Street Art', 'Fashion Styling', 'Vintage Collecting', 'Documentary Films', 'Historical Research', 'Urban Exploration', 'Adventure Travel', 'Wildlife Photography', 'Cultural Festivals', 'Food Photography', 'Plant-based Cooking', 'Home Brewing', 'Sustainable Fashion', 'Outdoor Survival Skills', 'Car Restoration', 'Extreme Sports', 'Off-Roading', 'Film Photography', 'Cocktail Mixology', 'Woodworking', 'DIY Electronics', 'Mobile Photography', 'Space Exploration', 'Stargazing', 'Meteorology', 'Fitness Coaching', 'Powerlifting', 'Bodybuilding', 'Yoga Instruction', 'Holistic Healing', 'Herbal Medicine', 'Tarot Card Reading', 'Witchcraft', 'Paranormal Investigations', 'Stand-up Comedy', 'Improvisational Theater', 'Political Activism', 'Human Rights Advocacy', 'Gender Studies', 'Intersectional Feminism', 'Community Gardening', 'Permaculture', 'Renewable Energy', 'Sustainable Architecture', 'Dance Choreography', 'Contemporary Art', 'Street Photography', 'Virtual Reality Gaming', 'App Development', 'Ethical Fashion', 'Social Entrepreneurship', 'Green Living', 'Zero Waste Lifestyle', 'Minimalism', 'Concert Photography', 'Musical Theatre', 'Voice Acting', 'Screenwriting', 'Historical Fiction', 'Astrophotography', 'Cryptography', 'Ethical Hacking', 'Podcasting', 'Language Learning', 'Calligraphy', 'Typography', 'Sketching', 'Comic Book Writing', 'Freelance Writing', 'Nature Conservation', 'Marine Biology', 'Birdwatching', 'Animal Rescue', 'Astronomy', 'Archaeology', 'Geology', 'Geography', 'Environmental Science', 'Human Anatomy', 'Classical Music', 'Opera', 'Street Dancing', 'Salsa Dancing', 'Fashion Illustration', 'Vintage Fashion', 'Culinary Tourism', 'Plant Care', 'Sustainable Gardening', 'Adventure Sports', 'Backpacking', 'Mountaineering', 'Parasailing', 'Kayaking', 'Sailing', 'Volunteer Work', 'Youth Mentoring', 'Nonprofit Organizations', 'Public Health', 'Nutrition', 'Life Coaching', 'Positive Psychology', 'Personal Development', 'Mindset Coaching', 'Meditation Retreats', 'Spiritual Healing', 'Holistic Nutrition', 'Eastern Philosophy', 'Ancient History', 'World Mythology', 'Fine Art Photography', 'Documentary Photography', 'Sustainable Travel', 'Cultural Anthropology', 'Linguistics', 'Language Translation', 'Robotics Engineering', 'Artificial Intelligence', 'Quantum Physics', 'Data Analytics', 'Virtual Assistant Development', 'Mobile App Design', 'Game Design', 'Bioengineering', 'Space Engineering', 'Renewable Energy Engineering', 'Cognitive Psychology', 'Neuropsychology', 'Philosophy of Mind', 'Environmental Ethics', 'Neuroethics', 'Bioethics', 'Social Psychology', 'Political Philosophy', 'Ethics in Technology', 'Science Fiction Writing', 'Fantasy Literature', 'Cultural Studies', 'Art History', 'Cosmology', 'Quantum Computing', 'Behavioral Economics', 'Market Research', 'Social Work', 'Community Development', 'Gender Equality', 'Inclusive Design', 'Environmental Activism', 'Sustainable Agriculture', 'Climate Change Mitigation''AI', 'Quantum Computing', 'Machine Learning', 'Philosophy', 'Virtual Reality', 'Augmented Reality', 'Robotics', 'Space Exploration', 'Astrophysics', 'Neuroscience', 'Cryptocurrency', 'Blockchain', 'Ethical Hacking', 'Cybersecurity', 'Data Science', 'Internet of Things', 'Bioinformatics', 'Genetics', 'Biotechnology', 'Renewable Energy', 'Sustainable Living', 'Environmental Conservation', 'Psychology', 'Cognitive Science', 'Artificial Life', 'Digital Art', 'Ethics', 'Consciousness Studies', 'Futurism', 'Transhumanism', 'Science Fiction', 'Cosmology', 'Nanotechnology', '3D Printing', 'Biohacking', 'Game Development', 'Mobile App Development', 'Web Development', 'Computer Graphics', 'Data Visualization', 'Human-Computer Interaction', 'Social Sciences', 'History', 'Anthropology', 'Sociology', 'Archaeology', 'Political Science', 'Economics', 'Philology', 'Linguistics', 'Literature', 'Creative Writing', 'Music', 'Film', 'Photography', 'Fine Arts', 'Graphic Design', 'Fashion Design', 'Culinary Arts', 'Interior Design', 'Architecture', 'Travel', 'Adventure', 'Sports', 'Fitness', 'Health and Wellness', 'Yoga', 'Meditation', 'Mindfulness', 'Cooking', 'Baking', 'Gardening', 'DIY Projects', 'Crafting', 'Collecting', 'Reading', 'Writing', 'Chess', 'Board Games', 'Puzzles', 'Hiking', 'Camping', 'Skiing', 'Snowboarding', 'Surfing', 'Scuba Diving', 'Rock Climbing', 'Mountaineering', 'Skydiving', 'Paragliding', 'Photography', 'Nature Conservation', 'Animal Welfare', 'Volunteering', 'Community Service', 'Humanitarian Work', 'Social Activism', 'Education', 'Teaching', 'Mentoring', 'Public Speaking', 'Debating', 'Philanthropy', 'Entrepreneurship', 'Startups', 'Investing', 'Personal Finance', 'Parenting', 'Family', 'Relationships', 'Self-improvement', 'Motivational Speaking', 'Coaching', 'Spirituality', 'Astrology', 'Tarot Reading', 'Yoga', 'Martial Arts', 'Dance', 'Music Production', 'DJing', 'Playing an Instrument', 'Singing', 'Film Making', 'Acting', 'Theater', 'Writing', 'Blogging', 'Podcasting', 'Social Media', 'Traveling', 'Exploring Different Cultures', 'Languages', 'Learning', 'Gaming', 'Cosplay', 'Comic Books', 'Anime', 'Manga', 'Fashion', 'Beauty', 'Food and Cooking', 'Coffee', 'Tea', 'Wine Tasting', 'Craft Beer', 'Fine Dining', 'Cuisine', 'Interior Design', 'Home Decor', 'DIY Home Improvement', 'Gardening', 'Outdoor Activities', 'Pet Care', 'Animal Training', 'Sustainability', 'Environmental Activism', 'Outdoor Photography', 'Fitness Training', 'Marathons', 'Triathlons', 'Cycling', 'Swimming', 'Team Sports', 'Adventure Races', 'Chess', 'Bridge', 'Poker', 'Billiards', 'Video Games', 'E-sports', 'Science Communication', 'Publications', 'Podcasts', 'Science Outreach''AI', 'NLP', 'Computer Vision', 'CV', 'Gaming', 'cooking']

    # Define a set of job keywords
    job_keywords = {'Accountant', 'Administrative Assistant', 'Software Engineer', 'Marketing Manager', 'Graphic Designer', 'Customer Service Representative', 'Project Manager', 'Sales Associate', 'Human Resources Specialist', 'Financial Analyst', 'Data Scientist', 'Business Development Manager', 'Operations Coordinator', 'Public Relations Coordinator', 'Event Planner', 'Content Writer', 'Operations Analyst', 'Quality Assurance Engineer', 'Social Media Specialist', 'Product Manager', 'IT Support Specialist', 'Legal Assistant', 'Research Analyst', 'Logistics Coordinator', 'Digital Marketing Manager', 'HR Generalist', 'Financial Controller', 'Technical Writer', 'Supply Chain Analyst', 'Operations Manager', 'Market Research Analyst', 'UX/UI Designer', 'Account Manager', 'Sales Executive', 'Data Entry Clerk', 'Marketing Coordinator', 'Medical Assistant', 'Web Designer', 'Dog Trainer', 'President', 'Software Developer', 'Data Analyst', 'Financial Advisor', 'Customer Support Specialist', 'Operations Manager', 'Business Analyst', 'Registered Nurse', 'Event Coordinator', 'Digital Marketing Specialist', 'IT Project Manager', 'Sales Representative', 'Content Manager', 'HR Manager', 'Operations Supervisor', 'Marketing Specialist', 'Executive Assistant', 'Front End Developer', 'Financial Planner', 'Customer Success Manager', 'Supply Chain Manager', 'Quality Control Inspector', 'Social Media Manager', 'Product Marketing Manager', 'Accounting Clerk', 'Administrative Coordinator', 'Database Administrator', 'Market Research Manager', 'UX Designer', 'Financial Analyst', 'Customer Service Manager', 'Business Development Coordinator', 'Operations Analyst', 'Public Relations Manager', 'Event Manager', 'Content Strategist', 'Data Scientist', 'IT Consultant', 'Legal Secretary', 'Research Assistant', 'Logistics Manager', 'Digital Marketing Coordinator', 'HR Assistant', 'Finance Manager', 'Technical Support Specialist', 'Supply Chain Coordinator', 'Operations Director', 'Market Analyst', 'UI Designer', 'Accounting Manager', 'Sales Manager', 'Data Analyst', 'Marketing Analyst', 'Medical Receptionist', 'Web Developer', 'Dog Groomer', 'CEO', 'Software Architect', 'Business Intelligence Analyst', 'Financial Controller', 'Customer Service Supervisor', 'Project Coordinator', 'Sales Coordinator', 'Content Writer', 'HR Coordinator', 'Operations Assistant', 'Marketing Assistant', 'Executive Secretary', 'Back End Developer', 'Investment Analyst', 'Customer Success Specialist', 'Supply Chain Specialist', 'Quality Assurance Analyst', 'Social Media Coordinator', 'Product Owner', 'Accounting Assistant', 'Administrative Secretary', 'Database Developer', 'Market Research Analyst', 'UX Researcher', 'Financial Advisor', 'Customer Support Representative', 'Operations Supervisor', 'Business Analyst', 'Registered Nurse', 'Event Planner', 'Digital Marketing Specialist', 'IT Project Coordinator', 'Sales Associate', 'Content Marketing Manager', 'HR Director', 'Operations Manager', 'Marketing Manager', 'Executive Director', 'Finance Analyst', 'Technical Writer', 'Supply Chain Manager', 'Quality Control Technician', 'Social Media Specialist', 'Product Manager', 'Account Executive', 'Sales Representative', 'Data Entry Specialist', 'Marketing Coordinator', 'Medical Secretary', 'Web Designer', 'Dog Walker', 'Chief Operating Officer', 'Software Engineer', 'Data Scientist', 'Financial Planner', 'Customer Support Manager', 'Supply Chain Analyst', 'Operations Analyst', 'Public Relations Specialist', 'Event Coordinator', 'Content Strategist', 'Data Analyst', 'IT Manager', 'Legal Assistant', 'Research Analyst', 'Logistics Coordinator', 'Digital Marketing Manager', 'HR Manager', 'Financial Controller', 'Technical Writer', 'Supply Chain Analyst', 'Operations Manager', 'Market Research Analyst', 'UX/UI Designer', 'developer', 'engineer', 'designer', 'manager', 'analyst', 'student'}

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
            if len(doc.ents) > 0 and doc.ents[0].label_ == "PERSON":
                user_name =doc.ents[0].text.capitalize()
                user_info['name'] = user_name
            else:
                tokenized_text = nltk.word_tokenize(user_input)
                for word in tokenized_text:
                    if word in names.words():
                        user_name = word.capitalize()
                        user_info['name'] = user_name


    for token in doc:
        # Check if the token text is in the set of hobby keywords
        if token.text.lower() in hobby_keywords:
            # Add the token as a hobby
            user_info['hobbies'].append(token.text)


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


        # Check if the token is in the list of interests
        if token.text.lower() == 'interested' and token.head.text == 'in':
            interest = token.head.head.text
            if interest.lower() in interests_keywords:
                user_info['interests'].append(interest)

        # Check if the token text is in the set of job keywords
        if token.text.lower() in job_keywords:
            # Add the token as a job
             user_info['jobs'].append(token.text)

    # Open the file for writing
    with open("output.txt", "w") as file:
        # Convert the dictionary to a JSON string
        user_info_str = json.dumps(user_info)

        # Write the JSON string to the file
        file.write(user_info_str)
    # Confirmation message
    print("User information saved to output.txt")
















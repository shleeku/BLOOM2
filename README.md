# BLOOM chatbot for VIP 701 

In "interact.py", change the following directory to choose where you want to save the model/weights:
data_dir="/mldata2/cache/transformers/bloom/"

and change this to specify which model/size you want to use:
checkpoint = "bigscience/bloom-7b1"


(Ignore this)
ssh-keygen -t rsa -b 4096 -C "email@example.com"
cat ~/.ssh/id_ed25519.pub
ssh-add ~/.ssh/id_ed25519
git remote add origin git@github.com:shleeku/BLOOM2.git
git push origin main

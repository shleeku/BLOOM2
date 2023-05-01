# BLOOM chatbot for VIP 701 

Environment:

Ubuntu 22.04.2

Pytorch 1.13.1

Trannsformers 4.26.0


In "interact.py", change the following directory to choose where you want to save the model/weights:

data_dir="/mldata2/cache/transformers/bloom/"

and change this to specify which model/size you want to use:

i.e.

checkpoint = "bigscience/bloom-7b1"

checkpoint = "bigscience/bloom-560m"


(Ignore the following)

ssh-keygen -t rsa -b 4096 -C "email@example.com"
cat ~/.ssh/id_ed25519.pub
ssh-add ~/.ssh/id_ed25519
git remote add origin git@github.com:shleeku/BLOOM2.git
git push origin main

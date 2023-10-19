# LLM chatbot for AI avatars

Environment:

Ubuntu 22.04.2

Pytorch 2.0.1

Transformers 4.29.2


In "interact_gguf.py", change the following directory to choose where you saved the model:

model_path="/mldata2/cache/transformers/llama2/llama-2-13b-chat.Q5_K_M.gguf"

Features:

Emotion recognition/generation  
-	확인 방법: 대화로 챗봇이 감정적인 반응 유도

User profile detection    
-	확인 방법: 대화하면서 이름, 나이 관심사 output.txt로 기록되는 것 확인

Matching algorithm  
-	확인 방법: 사용자 이름, 나이, 관심사 기록이 되면, 거기에 맞는 나이와 관심사 갖는 persona로 바꾸는 것 확인 (나이는 20세 이하, 20~40세, 40세 이상 되어야 하며 관심사는 완전 동일해야함)
	
Action generation  
-	확인 방법: 대화로 챗봇을 다음과 같은 말을 하도록 유도:  
-	인사 할 때 “handwave” 출력  
-	“yes”, “correct”,”nod 출력  
-	“no”, “wrong”,”headshake” 출력  
-	“congratulations”, “great!”,”clap” 출력  

챗봇 persona 설정  
-	확인 방법: 명렁어로 “let me talk to Brad”, “I want to speak to Skye” 등으로 persona 바꾸는 거 확인; context_brad.txt, context_jenny.txt 등에서 prompt 확인


(Ignore the following)

ssh-keygen -t rsa -b 4096 -C "email@example.com"
cat ~/.ssh/id_ed25519.pub
ssh-add ~/.ssh/id_ed25519
git remote add origin git@github.com:shleeku/BLOOM2.git
git push origin main

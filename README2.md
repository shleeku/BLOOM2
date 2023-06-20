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

If the GPU is not working try:

model = BloomForCausalLM.from_pretrained(checkpoint, cache_dir=data_dir, device_map="auto")



## Docker, Flask 이용

### interact_v2.py에서 수정할 부분
테스트할 땐 3번 GPU에서 사용했는데 데모 환경에 따라 35 line의 다음을 적절하게 수정
device = torch.device("cuda:3")

### Docker image build 이전에 nvidia toolkit 설치
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$(. /etc/os-release;echo $ID$VERSION_ID)/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get install -y nvidia-docker2

### 이후 Docker daemon reload
sudo pkill -SIGHUP dockerd

### /etc/docker/daemon.json파일 수정
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}

위를 복사해서 그대로 붙여넣기

### Docker 재시작
sudo systemctl restart docker


### Docker image 빌드: 
docker build -t chatbot_demo .

### 실행: 
docker run --runtime=nvidia --gpus all -p 5000:5000 -d chatbot_demo

### 정보 수신: 
curl -X POST -H "Content-Type: application/json" -d '{"User_ID": 000001, "message":"hello"}' http://localhost:5000/chat

정상 작동 시: {text: "Hi, how is it going?", action: "handwave", emotion: "joy"}
와 같은 JSON 형태의 output 송신


### Action generation
interact_flask.py 파일에서 prompting으로 action class generation하는 사례 확인가능.
BLOOM 176B 버전에서는 어느정도 성능이 나오는데 3B모델에서는 일관성이 없어서
본 데모에는 반영하지 못함.

실행방법: 
(Linux) export FLASK_APP=interact_flask.py

결과확인:
curl -X POST -H "Content-Type: application/json" -d '{"message":"hello"}' http://localhost:5000/chat

(Ignore the following)

ssh-keygen -t rsa -b 4096 -C "email@example.com"
cat ~/.ssh/id_ed25519.pub
ssh-add ~/.ssh/id_ed25519
git remote add origin git@github.com:shleeku/BLOOM2.git
git push origin main

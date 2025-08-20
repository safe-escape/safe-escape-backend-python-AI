실행방법
1. 로컬에서 uvicorn api:app --host 0.0.0.0 --port 8000 로 서버 실행.
2. ngork 사용해서 서버 외부 노출

(주의사항)
학습을 위해 crontab 사용해서 매시 정각마다 test.py와 train.py 실행해서 congestion.csv, density.csv 및 model 업데이트
1. crontab -e
2. 0 * * * * /usr/bin/python3(파이썬 절대경로) /home/ec2-user/test.py(test.py 절대경로) >> /home/ec2-user/cron.log(로그파일 절대경로) 2>&1
   5 * * * * /usr/bin/python3(파이썬 절대경로) /home/ec2-user/train.py(train.py 절대경로) >> /home/ec2-user/train.log(로그파일 절대경로) 2>&1 입력 후 :wq로 종료
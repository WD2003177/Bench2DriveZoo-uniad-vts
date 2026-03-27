docker run --gpus 'all,"capabilities=compute,video,utility"' --rm -it --network=host -v $(pwd)/code:/root/Bench2DriveZoo/team_code -v $(pwd)/ckpts:/root/Bench2DriveZoo/ckpts bench2drive /bin/bash

cd /root && SAVE_PATH=Bench2DriveZoo/team_code/output python Bench2DriveZoo/team_code/run.py

# FB15k-237
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 200 -g 60 \
  -lr 0.0005 --max_steps 100001 --cpu_num 1 --geo line --valid_steps 20000 \
  -linem "(1600,2)" -ql T

# NELL
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/NELL-betae -n 128 -b 512 -d 200 -g 60 \
  -lr 0.0005 --max_steps 100001 --cpu_num 1 --geo line --valid_steps 20000 \
  -linem "(1600,2)" -ql T

# WN18RR
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/WN18RR-betae -n 128 -b 512 -d 200 -g 60 \
  -lr 0.0005 --max_steps 100001 --cpu_num 1 --geo line --valid_steps 20000 \
  -linem "(1600,2)" -ql T

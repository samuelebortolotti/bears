python main.py --dataset kandinsky --model kanddpl --n_epochs 100 --lr 0.0005 --seed 0 \
--batch_size=64 --exp_decay=0.9  --c_sup 1 --w_c 10 \
--task patterns --warmup_steps 3 \
--posthoc --type resense --lambda_h 1  \
--checkin data/ckpts/kanddpl-150-entropy.pt \
--checkout \
--entropy --w_h 0.2
# --checkin data/ckpts/kanddpl-150-entropy.pt \

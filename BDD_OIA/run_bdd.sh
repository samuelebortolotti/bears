cd BDD_OIA; python main_bdd.py --train --h_type fcc --epochs 30 \
--batch_size 512 --nconcepts 30 --nconcepts_labeled 21 --h_sparsity 7 --opt adam --lr 0.005 \
--weight_decay 0.00004 --theta_reg_lambda 0.001 --obj ce \
--model_name dpl_auc --h_labeled_param 0 --w_entropy 1 --seed 42 # \
#--n-models 5 --do-test --deep_sep # --model_name dpl_auc
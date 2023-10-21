18.10.2023

./main.py --task=qmlp --model=352M --nb_train_samples=250000 --result_dir=results_qmlp_352M --batch_size=2

~11h per epoch on 3090 Ti

======================================================================
For the stack experiment:

./main.py --task=stack

Takes ~1h10min on a 4090.

======================================================================
For the arithmetic expressions experiments

# 38M parameters / 250k samples

./main.py --task=expr

# 352M parameters / 2.5M samples, reaches 99.80% after 12 epochs, the
  learning rate schedule is obviously terrible

./main.py --task=expr --nb_blocks=48 --dim_model=1024 --nb_train_samples=2500000 --result_dir=results_expr_48b_d1024_2.5M
======================================================================
25.07.2023

./main.py --task=sandbox --nb_train_samples=10000 --nb_test_samples=1000 --nb_blocks=4 --nb_heads=1 --nb_epochs=20

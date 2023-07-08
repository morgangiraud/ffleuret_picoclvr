
For the stack experiment:

./main.py --task=stack

Takes ~1h10min on a 4090.

For the arithmetic expressions experiments

# 38M parameters / 250k samples

./main.py --task=expr

# 352M parameters / 2.5M samples, reaches 99.80% after 12 epochs, the
  learning rate schedule is obviously terrible

./main.py --task=expr --nb_blocks=48 --dim_model=1024 --nb_train_samples=2500000 --result_dir=results_expr_48b_d1024_2.5M

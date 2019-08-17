(OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data challenger --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent separate > a_separate_challenger.out; OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data challenger --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent shared > a_shared_challenger.out)
(OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data fertility --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent separate > a_separate_fertility.out; OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data fertility --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent shared > a_shared_fertility.out)
(OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data concreteslump --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent separate > a_separate_concreteslump.out; OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data concreteslump --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent shared > a_shared_concreteslump.out)
(OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data servo --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent separate > a_separate_servo.out; OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data servo --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent shared > a_shared_servo.out)
(OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data machine --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent separate > a_separate_machine.out; OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data machine --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent shared > a_shared_machine.out)
(OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data yacht --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent separate > a_separate_yacht.out; OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data yacht --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent shared > a_shared_yacht.out)
(OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data autompg --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent separate > a_separate_autompg.out; OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data autompg --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent shared > a_shared_autompg.out)
(OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data housing --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent separate > a_separate_housing.out; OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python3 -u regression_runner.py --data housing --spacing random --ess_iters 10 --optim_iters 5 --iters 20 --mlatent shared > a_shared_housing.out)

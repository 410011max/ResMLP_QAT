# ResMLP_QAT
Quantization-aware training research for ResMLP model.


Finding initial scale for v2
```python
python quant_train.py -a q_resmlp_v2 --epochs 3 --lr 0.001 --batch-size 16 --data /mnt/disk1/imagenet/ --pretrained --save-path folder/checkpoints/ --wd 1e-4 --data-percentage 0.002 --gpu 1 --freeze-w --wandb

python quant_train.py -a q_resmlp --epochs 10 --lr 0.001 --batch-size 8 --data /dataset/imagenet --pretrained --save-path folder/checkpoints/ --wd 1e-4 --data-percentage 0.002 --gpu 0 --freeze-w --wandb

python quant_train.py -a q_trimmlp_12 --epochs 10 --lr 0.001 --batch-size 8 --data /dataset/imagenet --pretrained --save-path folder/checkpoints/ --wd 1e-4 --data-percentage 0.002 --gpu 0 --freeze-w --wandb
```
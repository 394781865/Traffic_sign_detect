Params:
	weight_decay: 0.0001
	bc_mode: True
	should_save_logs: True
	keep_prob: 0.8
	reduction: 0.5
	dataset: GTSRB
	model_type: DenseNet-BC
	depth: 40
	train: True
	should_save_model: True
	test: True
	renew_logs: True
	total_blocks: 3
	nesterov_momentum: 0.9
	growth_rate: 24
Train params:
	reduce_lr_epoch_1: 25
	initial_learning_rate: 0.1
	validation_split: None
	normalization: by_chanels
	reduce_lr_epoch_2: 50
	shuffle: every_epoch
	validation_set: False
	batch_size: 64
	n_epochs: 100

the version is for qinghua dataset add others category total 46 class
model = DenseNet(24,40,3,0.8,'DenseNet-BC',reduction=0.5,bc_mode=True)

import argparse

def str2bool(v):
	return v.lower() in ('yes', 'true', 't', '1', 'y')

def get_args():
	parser = argparse.ArgumentParser()
	parser.register('type', 'bool', str2bool)

	parser.add_argument('-debug',
			            type='bool',
						default=False,
						help='whether it is debug mode')

	parser.add_argument('-train',
			            type='bool',
						default=False,
						help='whether it is train mode')

	parser.add_argument('-log_file',
			            type=str,
						default=None,
						help='path of log file')

	parser.add_argument('-train_file',
			            type=str,
						default='../data/train.csv',
						help='folder of train data')

	parser.add_argument('-dev_file',
			            type=str,
						default='../data/dev.csv',
						help='path of dev file')

	parser.add_argument('-test_file',
			            type=str,
						default='../data/test.csv',
						help='path of test file')

	parser.add_argument('-result_file',
			            type=str,
						default='./result.csv',
						help='path of result file')

	parser.add_argument('-model_file',
			            type=str,
						default='../obj/S',
						help='path of model saved')

	# model para
	parser.add_argument('-batch_size',
			            type=int,
						default=32,
						help='size of batch')

	parser.add_argument('-N',
			            type=int,
						default=28,
						help='row size')

	parser.add_argument('-M',
			            type=int,
						default=28,
						help='column size')

	parser.add_argument('-lr',
			            type=float,
						default=0.01,
						help='learning rate')

	parser.add_argument('-epoch',
			            type=int,
						default=40,
						help='epoch size')

	parser.add_argument('-iter_cnt',
			            type=int,
						default=10,
						help='iteration count size')
	
	return parser.parse_args()

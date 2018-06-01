import utils
import config
import os
import json
import logging
import sys
import net
import torch
import torch.optim as optim
from torch.autograd import Variable

def save_model(model, epoch_id, path, prefix):
	torch.save({'epoch': epoch_id, 'model': model.state_dict()}, 
				os.path.join(path, '%sModel' % prefix))

def load_model(model, path, prefix):
	checkpoint = torch.load(os.path.join(path, prefix))
	model.load_state_dict(checkpoint['model'])

def test(dNet, input_x, pred_y):
	output_x = dNet(input_x)
	pred = output_x.data.max(1, keepdim=True)[1]
	correct += pred.eq(pred_y.data.view_as(pred)).sum()
	return float(correct) / len(dev_y)

def main(args):
	train_x, train_y = utils.load_data(args.train_file, args.N, args.M)
	dev_x, dev_y = utils.load_data(args.dev_file, args.N, args.M)
	logging.info('-' * 50)
	logging.info('Load train : %d, Load dev : %d' % (len(train_x), len(dev_x)))

	dNet = net.DigitRecNet()
	optimizer = optim.SGD(dNet.parameters(), lr=args.lr)
	criterion = torch.nn.NLLLoss()

	#train
	logging.info('-' * 50)
	logging.info('Start train ... ')

	dev_input_x = Variable(torch.FloatTensor(dev_input_x))
	dev_input_x = dev_input_x.resize(dev_input_x.size()[0], 1, args.N, args.M)
	dev_pred_y = Variable(torch.LongTensor(dev_pred_y))

	best_accuracy = 0
	for epoch_id in range(args.epoch):
		logging.info('Epoch : %d' % epoch_id)

		data = utils.random_data((train_x, train_y), args.batch_size)
		for it, (input_x, pred_y) in enumerate(data):
			input_x = Variable(torch.FloatTensor(input_x))
			input_x = input_x.resize(input_x.size()[0], 1, args.N, args.M)
			pred_y = Variable(torch.LongTensor(pred_y))
			assert input_x.size()[0] == pred_y.size()[0]

			optimizer.zero_grad()
			output_x = dNet(input_x)
			loss = criterion(output_x, pred_y)
			loss.backward()
			optimizer.step()

			logging.info('Iteration (%d) loss : %.6f' % (it, loss))

			if (it % args.iter_cnt == 0):
				tmp_accuracy = test(dNet, dev_input_x, dev_pred_y)
				if tmp_accuracy > best_accuracy:
					best_accuracy = tmp_accuracy
					save_model(dNet, epoch_id, args.model_file, 'BestModel')
				logging.info("Epoch : %d, Accuarcy : %.4f, Best Accuatcy : %.4f" 
							  % (epoch_id, tmp_accuracy, best_accuracy))

if __name__ == '__main__':
	args = config.get_args()

	if args.log_file is None:
		logging.basicConfig(level=logging.DEBUG,
							format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
	else:
		logging.basicConfig(filename=args.log_file,
							filemode='w', level=logging.DEBUG,
							format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

	logging.info(' '.join(sys.argv))
	logging.info(args)

	main(args)

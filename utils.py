import os
import json
import sys
import random

def load_image(data, N, M):
	assert len(data) == N * M
	image = []
	for i in range(N):
		row = []
		for j in range(M):
			row.append(int(data[i * M + j]))
		image.append(row)
	return image
	

def load_data(input_file, N, M):
	inf = open(input_file, 'r')
	Images = []
	Labels = []
	for line in inf:
		data = line[:-1].split(',')
		Images.append(load_image(data[1:], N, M))
		Labels.append(int(data[0]));
	return Images, Labels

def random_data(data, batch_size):
	length = len(data[0])
	idx = [x for x in range(length)]
	random.shuffle(idx)

	new_data = []
	for i in range(0, length, batch_size):
		tmpA = [data[0][x] for x in idx[i: min(i + batch_size, length)]]
		tmpB = [data[1][x] for x in idx[i: min(i + batch_size, length)]]
		new_data.append((tmpA, tmpB))
	return new_data

def expand(pred):
	length = len(pred)

	new_pred = []
	for i in range(length):
		new_pred.append([0 for x in range(10)])
		new_pred[-1][pred[i]] = 1
	return new_pred



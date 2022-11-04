"""
Author: Yunpeng Chen
"""
import math
import numpy as np
import cv2
import imageio
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt

class MGSampler(object):
	def __init__(self, num, mode):
		assert num > 0, "at least sampling 1 frame"
		self.num = num # clip_length
		self.mode = mode
	def multiplication(self, video_path):

		img = list()
		global count
		img_diff = list()
		img_diff.append(0.)

		try:
			vid = imageio.get_reader(video_path)
			for num, im in enumerate(vid):
				img.append(im)
			for i in range(len(img) - 1):
				tmp1 = cv2.cvtColor(img[i], cv2.COLOR_RGB2GRAY)
				tmp2 = cv2.cvtColor(img[i + 1], cv2.COLOR_RGB2GRAY)
				(score, diff) = compare_ssim(tmp1, tmp2, full=True)
				score = 1 - score

				img_diff.append(score)
			return img_diff

		except(OSError):
			print('error!')
			print(video_path.split('/')[-1])


	def sampling(self, video_path):

		def find_nearest(array, value):
			array = np.asarray(array)
			try:
				idx = (np.abs(array - value)).argmin()
				return int(idx + 1)
			except(ValueError):
				print('failed to find nearest')

		diff_score = self.multiplication(video_path)
		diff_score = np.power(diff_score, 2)
		sum_num = np.sum(diff_score)
		diff_score = diff_score / sum_num

		count_accu = 0
		pic_diff = list()
		for i in range(len(diff_score)):
			count_accu = count_accu + diff_score[i]
			pic_diff.append(count_accu)

		choose_index = list()
		if self.mode == 'train':
			for i in range(self.num):
				re = find_nearest(pic_diff, np.random.uniform((0+i)/self.num, (1+i)/self.num))
				re = re - 1
				choose_index.append(re)
		else:
			for i in range(self.num):
				re = find_nearest(pic_diff, (1/32) + (0+i)/self.num)
				re = re - 1
				choose_index.append(re)

		
		return choose_index


class RandomSampling(object):
	def __init__(self, num, interval=1, speed=[1.0, 1.0], seed=0):
		assert num > 0, "at least sampling 1 frame"
		self.num = num # clip_length
		self.interval = interval if type(interval) == list else [interval]
		self.speed = speed
		self.rng = np.random.RandomState(seed)

	def sampling(self, range_max, v_id=None, prev_failed=False):
		assert range_max > 0, \
			ValueError("range_max = {}".format(range_max))
		interval = self.rng.choice(self.interval)
		if self.num == 1:
			return [self.rng.choice(range(0, range_max))]
		# sampling
		speed_min = self.speed[0]
		speed_max = min(self.speed[1], (range_max-1)/((self.num-1)*interval))
		if speed_max < speed_min:
			return [self.rng.choice(range(0, range_max))] * self.num
		random_interval = self.rng.uniform(speed_min, speed_max) * interval
		frame_range = (self.num-1) * random_interval
		clip_start = self.rng.uniform(0, (range_max-1) - frame_range)
		clip_end = clip_start + frame_range
		return np.linspace(clip_start, clip_end, self.num).astype(dtype=np.int).tolist()


class SequentialSampling(object):
	def __init__(self, num, interval=1, shuffle=False, fix_cursor=False, seed=0):
		self.memory = {}
		self.num = num
		self.interval = interval if type(interval) == list else [interval]
		self.shuffle = shuffle
		self.fix_cursor = fix_cursor
		self.rng = np.random.RandomState(seed)

	def sampling(self, range_max, v_id, prev_failed=False):
		assert range_max > 0, \
			ValueError("range_max = {}".format(range_max))
		num = self.num
		interval = self.rng.choice(self.interval)
		frame_range = (num - 1) * interval + 1
		# sampling clips
		if v_id not in self.memory:
			clips = list(range(0, range_max-(frame_range-1), frame_range))
			if self.shuffle:
				self.rng.shuffle(clips)
			self.memory[v_id] = [-1, clips]
		# pickup a clip
		cursor, clips = self.memory[v_id]
		if not clips:
			return [self.rng.choice(range(0, range_max))] * num
		cursor = (cursor + 1) % len(clips)
		if prev_failed or not self.fix_cursor:
			self.memory[v_id][0] = cursor
		# sampling within clip
		idxs = range(clips[cursor], clips[cursor]+frame_range, interval)
		return idxs

class SegmentalSampling(object):
	def __init__(self, num_per_seg, segments=3, interval=1, shuffle=False, fix_cursor=False, seed=0):
		self.memory = {}
		self.num_per_seg = num_per_seg
		self.segments = segments
		self.interval = interval if type(interval) == list else [interval]
		self.shuffle = shuffle
		self.fix_cursor = fix_cursor
		self.rng = np.random.RandomState(seed)

	def sampling(self, range_max, v_id, prev_failed=False):
		assert range_max > 0, ValueError("range_max = {}".format(range_max))
		num_per_seg = self.num_per_seg 
		segments = self.segments	
		interval = self.rng.choice(self.interval)
		frame_range = (num_per_seg - 1) * interval + 1		

		idxs = []
		segment_length = range_max // segments
		for seg in range(segments):
			self.memory = {}
			start_id = int((seg) * segment_length)
			end_id = int((seg + 1) * segment_length - 1)

			# sampling clips
			if v_id not in self.memory:
				clips = list(range(start_id, end_id-(frame_range-1), frame_range))
				if self.shuffle:
					self.rng.shuffle(clips)
				self.memory[v_id] = [-1, clips]
			# pickup a clip
			cursor, clips = self.memory[v_id]
			if not clips:
				return [self.rng.choice(range(0, range_max))] * num_per_seg
			cursor = (cursor + 1) % len(clips)
			if prev_failed or not self.fix_cursor:
				self.memory[v_id][0] = cursor
			# sampling within clip
			idxs_seg = range(clips[cursor], clips[cursor]+frame_range, interval)
			idxs_seg = list(idxs_seg)
			for idx in idxs_seg:
				idxs.append(idx)

		return idxs



if __name__ == "__main__":

	import logging
	logging.getLogger().setLevel(logging.DEBUG)

	""" test RandomSampling() """
	# random_sampler = RandomSampling(num=16, interval=2, speed=[0.5, 1.0])

	# logging.info("RandomSampling(): range_max < num")
	# for i in range(10):
	# 	logging.info("{:d}: {}".format(i, random_sampler.sampling(range_max=2, v_id=1)))

	# logging.info("RandomSampling(): range_max == num")
	# for i in range(10):
	# 	logging.info("{:d}: {}".format(i, random_sampler.sampling(range_max=16, v_id=1)))

	# logging.info("RandomSampling(): range_max > num")
	# for i in range(10):
	# 	logging.info("{:d}: {}".format(i, random_sampler.sampling(range_max=30, v_id=1)))


	""" test SequentialSampling() """
	# sequential_sampler = SequentialSampling(num=3, interval=3, fix_cursor=False)

	# logging.info("SequentialSampling():")
	# for i in range(10):
	# 	logging.info("{:d}: v_id = {}: {}".format(i, 0, list(sequential_sampler.sampling(range_max=14, v_id=0))))
	# 	logging.info("{:d}: v_id = {}: {}".format(i, 1, sequential_sampler.sampling(range_max=9, v_id=1)))
	# 	logging.info("{:d}: v_id = {}: {}".format(i, 2, sequential_sampler.sampling(range_max=2, v_id=2)))
	# 	logging.info("{:d}: v_id = {}: {}".format(i, 3, sequential_sampler.sampling(range_max=3, v_id=3)))



	""" test SegmentalSampling() """
	# segmental_sampler = SegmentalSampling(num_per_seg=5, segments=3, interval=2, fix_cursor=False, shuffle=True)

	# logging.info("SegmentalSampling():")
	# for i in range(10):
	# 	logging.info("{:d}: v_id = {}: {}".format(i, 0, list(segmental_sampler.sampling(range_max=36, v_id=0))))
	""" test SegmentalSampling() """
	MG_Sampler = MGSampler(num=16)

	logging.info("MGSampler():")
	print(MG_Sampler.sampling(video_path='/media/mldadmin/home/s122mdg36_01/arid/assignment1/dataset/train_ying_caip/Pour/Pour_5_3.mp4'))
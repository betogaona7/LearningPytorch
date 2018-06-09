from __future__ import print_function, division
import os 
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings

warnings.filterwarnings('ignore') # Ignore warnings
plt.ion() # Interactiv mode 

# Show image with landmarks
def show_landmarks(image, landmarks):
	plt.imshow(image)
	plt.scatter(landmarks[:,0], landmarks[:,1], s=10, marker='.', c='r')
	plt.pause(5) # Pause a bit so that plots are updated

def show landmarks_batch(sampled_batched):
	""" 
		Show image with landmarks for a batch of samples
	"""
	images_batch, landmarks_batch = sampled_batched['image'], sampled_batched['landmarks']
	batch_size = len(image_batch)
	im_size = images_batch.size(2)
	grid = utils.make_grid(images_batch)
	plt.imshow(grid.numpy().transpose((1, 2, 0)))
    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')
        plt.title('Batch from dataloader')

# READ THE CVS
landmarks_fram = pd.read_csv('./data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_fram.iloc[n, 0]
landmarks = landmarks_fram.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)), landmarks)
plt.show()


class FaceLandmarksDataset(Dataset):
	def __init__(self, csv_file, root_dir, transform=None):
		self.landmarks_frame = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self): # Returns the size of the dataset
		return len(self.landmarks_frame)

	def __getitem__(self, idx): # Suppoort indexing such that dataset [i] can be used to get ith sample
		img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
		image = io.imread(img_name)
		landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix() 
		landmarks = landmarks.astype('float').reshape(-1, 2)
		sample = {'image':image, 'landmarks': landmarks}

		if self.transform:
			sample = self.transform(sample)
		return sample

# Read the cvs in __init__ but leave the reading of images to __getitem__ is memory efficient because
# all the images are not stored in the memory at once but read as required

# Iterate through data samples
face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv', root_dir='data/faces/')
fig = plt.figure() 
for i in range(len(face_dataset)):
	sample = face_dataset[i]
	print(i, sample['image'].shape, sample['landmarks'].shape)
	ax = plt.subplot(1, 4, i+1)
	plt.tight_layout()
	ax.set_title('Sample #()'.format(i))
	ax.axis('off')
	show_landmarks(**sample)
	if i == 3:
		plt.show()
		break

# Preprocessing 
class Rescale(object):
	""" 
		Rescale the image in a sample to a given size
		Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
	"""
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, landmarks = sample['image'], sample['landmarks']
		h, w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = self.output_size
		new_h, new_w = int(new_h), int(new_w)
		img = transform.resize(image, (new_h, new_w))
		# h and w are swapped for landmarks because for images x and y axes are axis 1 and 0 respectively
		landmarks = landmarks * [new_w/w, new_h/h]
		return {'image':img, 'landmarks':landmarks}

class RandomCrop(object): # Data augmentation
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h,
                      left: left + new_w]
        landmarks = landmarks - [left, top]
        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because numpy image is: H x W x C and torch image is: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


# Apply transform on a sample
scale = Rescale(256) #  Rescale the shorter side of the image to 256
crop = RandomCrop(128) # Then randomly crop a square of size 224 from it
composed = transforms.Compose([Rescale(256), RandomCrop(224)]) # Compose rescale and random crop transforms

fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
	transformed_sample = tsfrm(sample)
	ax = plt.subplot(1, 3, i+1)
	plt.tight_layout()
	ax.set_title(type(tsfrm).__name__)
	show_landmarks(**transformed_sample)
plt.show()

# iteration through the dataset 
# Every time this dataset is sampled: 
# 	- An image is read from the file on fly
# 	- Tranforms are applied on the read image
#	- Since one of the tranforms is random, data augmentated on sampling
transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
										   root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))
# In this way we lose a lot of features
# - Batching the data
# - Shuffling the data
# - Load the data in parallel using multiprocessing workers
for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['landmarks'].size())
    if i == 3:
        break

# Dataloader is an iterator which provides the features missed in the for loop.
# collate_fn, specify how exactly the samples need to be batched
dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
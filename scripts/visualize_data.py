import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d
from ml3d.vis import Visualizer, LabelLUT

# construct a dataset by specifying dataset_path
dataset = ml3d.datasets.AHAT(dataset_path='/Users/kathyxxj/Documents/mrlab/dataset/AHAT/')

# get the 'all' split that combines training, validation and test set
all_split = dataset.get_split('train')

# print the attributes of the first datum
print(all_split.get_attr(0))

# print the shape of the first point cloud
print(all_split.get_data(0)['point'].shape)

# show the first 100 frames using the visualizer
vis = ml3d.vis.Visualizer()
ahat_labels = ml3d.datasets.AHAT.get_label_to_names()
lut = LabelLUT()
for val in sorted(ahat_labels.keys()):
    lut.add_label(ahat_labels[val], val)
vis.set_lut("labels", lut)
vis.visualize_dataset(dataset, 'train')
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pathlib
import random
import os

# import OrganizeTrainImages
# import SortImages
# import Convert_to_jpeg

sample_buffer_size = 128

# set up dataset
# os.system("python Convert_to_jpeg.py")    #not currently needed
# os.system("python OrganizeTrainImages.py")
# os.system("python SortImages.py")


# some settings
BATCH_SIZE = 64
tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

# downloads file from internet
# data_root = tf.keras.utils.get_file('flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)
# print("data root is: "+data_root)

# version for picnic hack
data_root = r"C:\Users\jesus\Desktop\The Picnic Hackathon 2019 - Copy"

data_root = pathlib.Path(data_root)

# gets all image paths, then turns them into list of str
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]

random.shuffle(all_image_paths)

# image_count = len(all_image_paths)
image_count = sample_buffer_size

# attributions = (data_root/"LICENSE.txt").read_text(encoding="utf8").splitlines()[4:]
# attributions = [line.split(' CC-BY') for line in attributions]
# attributions = dict(attributions)

# displays images
'''
for n in range(3):
  image_path = random.choice(all_image_paths)
  display.display(display.Image(image_path))
  print(image_path)
  img=mpimg.imread(image_path)
  plt.figure()
  plt.imshow(img)
  plt.show()
  '''

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]
print("First 10 labels indices: ", all_image_labels[:10])

img_path = all_image_paths[0]
print(img_path)
img_raw = tf.read_file(img_path)
img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape)
print(img_tensor.dtype)

img_final = tf.image.resize_images(img_tensor, [192, 192])
img_final = img_final / 255.0
print(img_final.shape)


# fix for jpeg
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize_images(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


def change_range(image, label):
    return 2 * image - 1, label


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

# image and labels dataset
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

# replace for tfrecords?
image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.read_file)
tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(image_ds)
image_ds = tf.data.TFRecordDataset('images.tfrec').map(preprocess_image)
ds = tf.data.Dataset.zip((image_ds, label_ds))
ds = ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)

ds = image_label_ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable = False

keras_ds = ds.map(change_range)
image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)
model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names))])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

model.fit(ds, epochs=63, steps_per_epoch=10)

model.save("myModel.h5")

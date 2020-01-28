import tensorflow as tf
import pathlib
import random

#types are 0-24



def createImageData():
    data_root = r"C:\Users\jesus\Desktop\The Picnic Hackathon 2019 - Copy"
    data_root = pathlib.Path(data_root)

    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    print(all_image_paths[:10])

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    print(label_names[:10])

    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]

    c=0
    all_instance_ids=[]
    for i in all_image_labels:
        all_instance_ids.append(c)
        c+=1
    return all_image_paths,all_image_labels,all_instance_ids



#createImageData()


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def write_examples(image_data, output_path):
    """
    Create a tfrecord file.

    Args:
      image_data (List[(image_file_path (str), label (int), instance_id (str)]): the data to store in the tfrecord file.
        The `image_file_path` should be the full path to the image, accessible by the machine that will be running the
        TensorFlow network. The `label` should be an integer in the range [0, number_of_classes). `instance_id` should be
        some unique identifier for this example (such as a database identifier).
      output_path (str): the full path name for the tfrecord file.
    """
    writer = tf.python_io.TFRecordWriter(output_path)

    for image_path, label, instance_id in image_data:
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'label': _int64_feature([label]),
                'path': _bytes_feature([image_path]),
                'instance': _bytes_feature([instance_id])
            }
        ))

        writer.write(example.SerializeToString())

    writer.close()

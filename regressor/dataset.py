from tensorflow.data import TFRecordDataset
import tensorflow as tf
import os
import glob

def create_dataset_from_tfrecords(train=True):
    """Creates dataset from tfrecords"""
    if train:
        files = glob.glob(f"{os.getenv('DATA_DIR')}tfrecord/train_*.tfrecord")
    else:
        files = glob.glob(f"{os.getenv('DATA_DIR')}tfrecord/eval*.tfrecord")

    raw_dataset = TFRecordDataset(files)
    return raw_dataset

def _parse_function(serialized_example):
    """
    Parse a single serialized tf.train.Example.

    Args:
        serialized_example (tf.Tensor): A serialized tf.train.Example.

    Returns:
        tuple: A tuple containing the parsed example and sequence length.
    """
    feature_description = {
        "encseq": tf.io.VarLenFeature(tf.int64),
        "charge": tf.io.FixedLenFeature([], tf.int64),
        "modified_sequence": tf.io.FixedLenFeature([], tf.string),
        "norm_mz": tf.io.FixedLenFeature([], tf.float32),
        "target_1": tf.io.FixedLenFeature([], tf.float32),
        "target_2": tf.io.FixedLenFeature([], tf.float32),
        "length": tf.io.FixedLenFeature([], tf.int64),
        "metadata": tf.io.FixedLenFeature([2], tf.float32),
    }

    # Parse the example
    parsed_example = tf.io.parse_single_example(serialized_example, feature_description)

    # Decode string feature
    parsed_example["modified_sequence"] = tf.io.decode_raw(parsed_example["modified_sequence"], tf.uint8)

    # Convert VarLenFeature to Dense tensor
    parsed_example["encseq"] = tf.sparse.to_dense(parsed_example["encseq"], default_value=0)
    #cast to int32
    parsed_example["encseq"] = tf.cast(parsed_example["encseq"], tf.int32)
    parsed_example["charge"] = tf.cast(parsed_example["charge"], tf.int32)


    # keep only encseq, metadata, and target_1
    target_1 = parsed_example["target_1"]
    parsed_example = {
        "encseq": parsed_example["encseq"],
        "charge": parsed_example["charge"],
    }

    return parsed_example, target_1

def process_training_set(num_epochs, batch_size):
    """
    Create training set from tfrecords and process it.

    Args:
        num_epochs (int): Number of epochs to repeat the dataset.
        batch_size (int): Batch size.
    
    Returns:
        tf.data.Dataset: Processed training set.
    """

    train_set = create_dataset_from_tfrecords(train=True)
    train_set = train_set.map(_parse_function)
    train_set = train_set.shuffle(10000)
    train_set = train_set.repeat(num_epochs)
    #pad the sequences to the same length
    train_set = train_set.padded_batch(batch_size=batch_size,
                                       padded_shapes=({'encseq': [None], 'charge': []}, []),
                                       padding_values=({'encseq': 0, 'charge': 0}, 0.0)
                                       )
    return train_set

def process_eval_set(batch_size):
    """
    Create eval set from tfrecords and process it.

    Args:
        batch_size (int): Batch size.
    
    Returns:
        tf.data.Dataset: Processed eval set.
    """

    eval_set = create_dataset_from_tfrecords(train=False)
    eval_set = eval_set.map(_parse_function)
    #pad the sequences to the same length
    eval_set = eval_set.padded_batch(batch_size=batch_size,
                                     padded_shapes=({'encseq': [None], 'charge': []}, []),
                                     padding_values=({'encseq': 0, 'charge': 0}, 0.0)
                                     )
    return eval_set
                                      
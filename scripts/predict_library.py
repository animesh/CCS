#%%
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
sys.path.append("../")
from regressor_tf.model_rnn import BIRNN
from scripts.utils import encoded_sequence, int_dataset
import tf2onnx
import onnx

os.environ["CUDA_VISIBLE_DEVICES"]="0"
def set_memory_growth():
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

def load_model(model_path, model_params):
    model = BIRNN(**model_params)
    model.build(input_shape=[(None, 66), (None, 1)])
    model.load_weights(model_path)
    return model

def load_data(data_path):
    return pd.read_csv(data_path, sep="\t")

def create_dataset(data):
    data["encseq"] = encoded_sequence(data)
    timesteps = 66
    features = int_dataset(data, timesteps, middle=False).squeeze()
    labels = np.zeros(features.shape[0])
        
    def test_generator():
        for i in range(len(features)):
            yield (features[i,:-1], features[i,-1]), labels[i]

    types = ((tf.int32, tf.int32), tf.float32)
    shapes = (([66], ()), ())
    dataset_test = tf.data.Dataset.from_generator(test_generator, output_types=types, output_shapes=shapes).batch(512)
    return dataset_test

def predict(model, dataset_test):
    return model.predict(dataset_test).squeeze()

def sort_values(data, column):
    return data[column].sort_values(ascending=True)

def check_equality(data1, data2):
    return data1.sort_values(ascending=True).eq(data2.sort_values(ascending=True)).all()

def get_max_id(msms):
    return msms["id"].max()

def create_new_msms(msms, id_max):
    msms_newids = msms.copy()
    new_ids = msms["id"].values + id_max
    msms_newids["id"] = new_ids
    return msms_newids

def create_new_evidence(evidence, id_max):
    evidence_newids = evidence.copy()
    new_ids = evidence["MS/MS IDs"].values + id_max
    evidence_newids["MS/MS IDs"] = new_ids
    return evidence_newids

def concatenate_data(data1, data2):
    return pd.concat([data1, data2], axis=0)

def add_predictions(data, predictions, column_name):
    data[column_name] = predictions
    return data

def drop_column(data, column):
    data.drop(columns=column, inplace=True)
    return data

def save_data(data, output_file):
    #rename file to avoid overwriting
    data.to_csv(output_file, sep="\t", index=False)

def save_model_as_pb(model_path, model_params, export_dir):
    graph = tf.Graph()
    with tf.compat.v1.Session(graph=graph) as sess:
        # Restore the model from the model_path
        model = load_model(model_path, model_params)
        # Save the model as a SavedModel
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.SERVING])
        builder.save()

def save_model_as_onnx(model_path, model_params, export_path):
    model = load_model(model_path, model_params)
    # Convert the model to ONNX
    input_shape = [(None, 66), (None, 1)]
    input_signature = [tf.TensorSpec(shape=shape, dtype=tf.float32) for shape in input_shape]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)
    onnx.save_model(onnx_model, export_path)

def generate_library_two_vaues(paths_to_models, output_paths):

    #check that paths_to_models is a dictionary
    if not isinstance(paths_to_models, dict):
        raise ValueError("paths_to_models must be a dictionary")
    #check that output_paths is a dictionary
    if not isinstance(output_paths, dict):
        raise ValueError("output_paths must be a dictionary")
    
    set_memory_growth()

    model_params = {
        "num_hidden" : 128, "num_layers" : 2, "num_classes" : 1,
        "embed_dim" : 66, "dict_size" : 32, "dropout_rate" : 0.4
    }

    model_dict = {}

    print("loading models...")
    for population in ["upper", "lower"]:
        model_dict[population] = load_model(paths_to_models[population], model_params)

    data_path = "/fs/pool/pool-cox-data08/Juan/ccs/Data/test/libraries/homo_sapiens/homo_sapiens_evidence.txt"
    evidence = load_data(data_path)

    print("Creating dataset...")
    dataset_test = create_dataset(evidence)

    print("predicting with upper model")
    y_pred_upper = predict(model_dict["upper"], dataset_test)
    print("predicting with lower model")
    y_pred_lower = predict(model_dict["lower"], dataset_test)

    print("process msms")
    msms_path = "/fs/pool/pool-cox-data08/Juan/ccs/Data/test/libraries/homo_sapiens/homo_sapiens_msms.txt"
    msms = load_data(msms_path)

    id_max = get_max_id(msms)
    msms_newids = create_new_msms(msms, id_max)
    msms_double = concatenate_data(msms, msms_newids)
    #remove individual elements
    del msms, msms_newids

    print("process evidence")
    evidence_newids = create_new_evidence(evidence, id_max)
    evidence = add_predictions(evidence, y_pred_lower, "Calibrated 1/K0")
    evidence_newids = add_predictions(evidence_newids, y_pred_upper, "Calibrated 1/K0")
    evidence_double = concatenate_data(evidence, evidence_newids)
    #remove individual elements
    del evidence, evidence_newids
    evidence_double = drop_column(evidence_double, "encseq")

    print("saving data")
    save_data(evidence_double, output_paths["evidence"])
    save_data(msms_double, output_paths["msms"])

def generate_library_single_value(path_to_model, output_paths):

    #check that paths are strings
    if not isinstance(path_to_model, str):
        raise ValueError("path must be a string")
    #check that output_paths is a dictionary
    if not isinstance(output_paths, dict):
        raise ValueError("output_paths must be a dictionary")
    
    set_memory_growth()

    model_params = {
        "num_hidden" : 128, "num_layers" : 2, "num_classes" : 1,
        "embed_dim" : 66, "dict_size" : 32, "dropout_rate" : 0.2
    }

    model = load_model(path_to_model, model_params)

    print("loading evidence")
    data_path = "/fs/pool/pool-cox-projects-tesorai/hela_astral_manuscript/results/fdr_001/hela_astral_1da_15_min/1/combined/libs/g0/evidence.txt"
    evidence = load_data(data_path)

    print("Creating dataset...")
    dataset_test = create_dataset(evidence)

    print("predicting with upper model")
    y_pred = predict(model, dataset_test)

    print("process msms")
    msms_path = "/fs/pool/pool-cox-projects-tesorai/hela_astral_manuscript/results/fdr_001/hela_astral_1da_15_min/1/combined/libs/g0/msms.txt"
    msms = load_data(msms_path)

    print("process evidence")

    evidence = drop_column(evidence, "encseq")
    evidence = drop_column(evidence, "Retention time")

    evidence = add_predictions(evidence, y_pred, "Retention time")

    print("saving data")
    save_data(evidence, output_paths["evidence"])
    save_data(msms, output_paths["msms"])
#%%
if __name__ == "__main__" :
    model_path = "/home/rlopez/Documents/rt/model/checkpoints/best"
    output_paths = {
        "evidence" : "/fs/pool/pool-cox-data08/Juan/libs_for_shamil/evidence_newrt.txt",
        "msms" : "/fs/pool/pool-cox-data08/Juan/libs_for_shamil/msms_newrt.txt"
    }
    generate_library_single_value(model_path, output_paths=output_paths)
# %%

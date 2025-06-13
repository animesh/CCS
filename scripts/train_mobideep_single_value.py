import pandas as pd
import tensorflow as tf
import os
from utils import encoded_sequence, int_dataset
from sklearn.model_selection import train_test_split
import sys
sys.path.append("../regressor_tf")
from model_rnn import BIRNN

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
    def get_config(self):
        config = {
        'd_model': self.d_model,
        'warmup_steps': self.warmup_steps,
         }
        return config

#select cuda device
os.environ["CUDA_VISIBLE_DEVICES"]="0"

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

root_path = "/fs/pool/pool-cox-data08/Juan/ccs/Data/"
data = pd.read_pickle(os.path.join(root_path, "two_populations", "evidence_aligned_train_v2660.pkl"))

mask = (data["Reverse"] != "+") & (data["Charge"] > 1) & (data["Intensity"] > 0 & (~data["Aligned 1/K0"].isna()) )
data = data[mask]

data_most_intense = data.sort_values(by=["Modified sequence", "Charge", "Intensity"], ascending=False)
#group by modified sequence, charge and take the first row
data_most_intense = data_most_intense.groupby(["Modified sequence", "Charge"]).first().reset_index()

data_most_intense["encseq"] = encoded_sequence(data_most_intense)

timesteps = 66
features = int_dataset(data_most_intense, timesteps, middle=False).squeeze()
labels = data_most_intense["Aligned 1/K0"].values

X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, random_state=42)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

#Function to create a generator for the training data
def train_generator():
    for i in range(len(X_train)):
        yield (X_train[i,:-1], X_train[i,-1]), y_train[i]

#Function to create a generator for the validation data
def eval_generator():
    for i in range(len(X_val)):
        yield (X_val[i,:-1], X_val[i,-1]), y_val[i]

# Define the types and shapes for your dataset
# Assuming encseq is a variable-length list, charge is a scalar integer, and labels are strings
types = ((tf.int32, tf.int32), tf.float32)
shapes = (([66], ()), ())

# Create the TensorFlow dataset
dataset_train = tf.data.Dataset.from_generator(train_generator, output_types=types, output_shapes=shapes)

# Create the TensorFlow dataset
dataset_eval = tf.data.Dataset.from_generator(eval_generator, output_types=types, output_shapes=shapes)

model_params = {
    "num_hidden" : 128,
    "num_layers" : 2,
    "num_classes" : 1,
    "embed_dim" : 66,
    "dict_size" : 32,
    "dropout_rate" : 0.4,  
}
model = BIRNN(**model_params)
model.build(input_shape=[(None, 66), (None, 1)])
model.summary()

training_parameters = {
"batch_size": 64,
"batch_size_eval": 128,
"num_epochs": 200,
"learning_rate_rnn": 5e-5,#5e-4 is already too big, if anything make it smaller
"num_warmup_steps": 10000,#3 epochs
"learning_rate_transformer": 4e-5,
}

dataset_train = dataset_train.shuffle(1000).repeat(None).batch(training_parameters["batch_size"])
dataset_eval = dataset_eval.batch(training_parameters["batch_size_eval"])

scheduler_rnn = CustomSchedule(model_params["embed_dim"]*16, training_parameters["num_warmup_steps"])
optimizer_rnn = tf.keras.optimizers.Adam(scheduler_rnn, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

training_batches = len(X_train) // training_parameters["batch_size"]
eval_batches = len(X_val) // training_parameters["batch_size_eval"]

class LearningRateCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        print(f"Current learning rate: {lr:.5f}")

path = "/fs/pool/pool-cox-data08/Juan/ccs/models/birnn_rpr"
cb = [tf.keras.callbacks.CSVLogger(f'{path}/training.log', append=False),  
    tf.keras.callbacks.ModelCheckpoint(f'{path}/checkpoints/best', save_best_only=True, save_weights_only=True),
    LearningRateCallback()]

model.compile(loss='mean_squared_error', optimizer=optimizer_rnn)

history = model.fit(
    dataset_train, epochs=training_parameters["num_epochs"], validation_data=dataset_eval,
    steps_per_epoch=training_batches, validation_steps=eval_batches, callbacks=cb)




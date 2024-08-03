import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from inference import Inference
from trainer import ClassificationTrianer
from dataloader.dataloader import DigitImageDataset
from models.DigitRecognizerModel import (
    DigitRecognizerLargeCNNModel,
    DigitRecognizerMLPModel,
    DigitRecognizerCNNModel,
)

num_epochs = 20
training_csvfile = os.path.join("data", "train.csv")
testing_csvfile = os.path.join("data", "test.csv")

training_dataset = DigitImageDataset(csv_file=training_csvfile, validation=False)
validation_dataset = DigitImageDataset(csv_file=training_csvfile, validation=True)
testing_dataset = DigitImageDataset(
    csv_file=testing_csvfile, testset=True, shuffle=False
)

batch_size = 128
training_dataloader = DataLoader(
    training_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
testing_dataloader = DataLoader(testing_dataset, batch_size=1)

mlp_model = DigitRecognizerMLPModel()
cnn_model = DigitRecognizerCNNModel()
large_cnn_model = DigitRecognizerLargeCNNModel()

learning_rate = 0.001

mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)
large_cnn_optimizer = optim.Adam(large_cnn_model.parameters(), lr=learning_rate)

loss_function = nn.NLLLoss()

# Try the MLP Model
mlp_model_dir = "MLP_checkpoints"
save_frequency = 5
mlp_trainer = ClassificationTrianer(
    model=mlp_model,
    optimizer=mlp_optimizer,
    loss_function=loss_function,
    training_dataloader=training_dataloader,
    validation_dataloader=validation_dataloader,
    num_epochs=num_epochs,
    save_frequency=save_frequency,
    device="cuda",
    model_dir=mlp_model_dir,
    load_model=False,
)
mlp_accuracy = mlp_trainer.train()

# Try the CNN Model
cnn_model_dir = "CNN_checkpoints"
cnn_trainer = ClassificationTrianer(
    model=cnn_model,
    optimizer=cnn_optimizer,
    loss_function=loss_function,
    training_dataloader=training_dataloader,
    validation_dataloader=validation_dataloader,
    num_epochs=num_epochs,
    save_frequency=save_frequency,
    device="cuda",
    model_dir=cnn_model_dir,
    load_model=False,
)

cnn_accuracy = cnn_trainer.train()

# Try the Large CNN Model
large_cnn_model_dir = "Large_CNN_checkpoints"
large_cnn_trainer = ClassificationTrianer(
    model=large_cnn_model,
    optimizer=large_cnn_optimizer,
    loss_function=loss_function,
    training_dataloader=training_dataloader,
    validation_dataloader=validation_dataloader,
    num_epochs=num_epochs,
    save_frequency=save_frequency,
    device="cuda",
    model_dir=large_cnn_model_dir,
    load_model=False,
)

large_cnn_accuracy = large_cnn_trainer.train()

print(
    f"MLP Accuracy: {mlp_accuracy*100}%, CNN Accuracy: {cnn_accuracy*100}% and Large CNN Accuracy: {large_cnn_accuracy*100}%"
)

model_dict = {
    mlp_accuracy: (mlp_model, mlp_model_dir),
    cnn_accuracy: (cnn_model, cnn_model_dir),
    large_cnn_accuracy: (large_cnn_model, large_cnn_model_dir),
}
model, model_dir = model_dict[max(model_dict.keys())]

# Inference Model
inference_object = Inference(
    model=model,
    testing_dataloader=testing_dataloader,
    device="cuda",
    load_model=True,
    load_model_path=os.path.join(model_dir, f"checkpoint_{num_epochs}.pt"),
    output_file="final_submission.csv",
)

inference_object.make_inference()

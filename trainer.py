import os
import torch
from tqdm import tqdm
from torch.optim import optimizer


class ClassificationTrianer:
    def __init__(
        self,
        model,
        optimizer,
        loss_function,
        training_dataloader,
        validation_dataloader,
        num_epochs=100,
        save_frequency=25,
        device="cuda",
        model_dir="checkpoints",
        load_model=False,
        load_model_path="",
    ):
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        self.model = (
            model.to(self.device)
            if model is not None
            else self.raise_exception("Model is None")
        )
        self.optimizer = (
            optimizer
            if optimizer is not None
            else self.raise_exception("Optimizer is None")
        )
        self.loss_function = (
            loss_function.to(self.device)
            if loss_function is not None
            else self.raise_exception("Loss Function is None")
        )
        self.training_dataloader = (
            training_dataloader
            if training_dataloader is not None
            else self.raise_exception("Training Dataloader is None")
        )
        self.validation_dataloader = (
            validation_dataloader
            if validation_dataloader is not None
            else self.raise_exception("Validation Dataloader is None")
        )
        self.num_epochs = num_epochs
        self.save_frequency = save_frequency
        self.model_dir = model_dir
        if load_model == True:
            if os.path.exists(load_model_path):
                self.load_model(load_model_path)
            else:
                self.raise_exception("Load Model path is empty")

    def raise_exception(self, message):
        raise Exception(message)

    def train_one_epoch(self):
        average_loss = 0
        average_accuracy = 0
        for label, image in self.training_dataloader:
            self.optimizer.zero_grad()
            label = label.to(self.device)
            image = image.to(self.device)
            output = self.model(image)
            output_label = torch.argmax(output, dim=1)
            accuracy = (output_label == label).float().mean()
            loss = self.loss_function(output, label)
            loss.backward()
            self.optimizer.step()
            average_loss += loss.item()
            average_accuracy += accuracy
        average_accuracy = average_accuracy / len(self.training_dataloader)
        average_loss = average_loss / len(self.training_dataloader)

        return average_loss, average_accuracy

    def train(self):
        final_accuracy = 0
        progress_bar = tqdm(range(self.num_epochs))
        for epoch in progress_bar:
            self.model.train()
            train_loss, train_accuracy = self.train_one_epoch()
            validation_loss, validation_accuracy = self.validate()
            final_accuracy = validation_accuracy
            progress_bar.set_description(
                f"Epoch: {epoch+1}/{self.num_epochs} Training Loss: {train_loss} "
                + f"Validation Loss: {validation_loss} Training Accuracy: {train_accuracy} "
                + f"Validation Accuracy {validation_accuracy}"
            )
            if (epoch + 1) % self.save_frequency == 0:
                self.save_model(epoch + 1)

        return final_accuracy

    def validate(self):
        average_loss = 0
        average_accuracy = 0
        self.model.eval()
        for label, image in self.validation_dataloader:
            label = label.to(self.device)
            image = image.to(self.device)
            output = self.model(image)
            output_label = torch.argmax(output, dim=1)
            accuracy = (output_label == label).float().mean()
            loss = self.loss_function(output, label)
            average_loss += loss.item()
            average_accuracy += accuracy
        average_accuracy = average_accuracy / len(self.validation_dataloader)
        average_loss = average_loss / len(self.validation_dataloader)

        return average_loss, average_accuracy

    def save_model(self, epoch):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, f"checkpoint_{epoch}.pt"),
        )

    def load_model(self, model_path):
        print(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path))

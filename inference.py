import os
import torch
from tqdm import tqdm


class Inference:
    def __init__(
        self,
        model,
        testing_dataloader,
        device="cuda",
        load_model=True,
        load_model_path="",
        output_file="submission.csv",
    ):
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        self.model = (
            model.to(self.device)
            if model is not None
            else self.raise_exception("Model is None")
        )
        self.testing_dataloader = (
            testing_dataloader
            if testing_dataloader is not None
            else self.raise_exception("Testing Dataloader is None")
        )
        self.output_file = output_file
        if load_model == True:
            if os.path.exists(load_model_path):
                self.load_model(load_model_path)
            else:
                self.raise_exception("Load Model path is empty")

    def raise_exception(self, message):
        raise Exception(message)

    def load_model(self, model_path):
        print(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path))

    def inference(self):
        self.model.eval()
        outputs = {}
        for i, image in enumerate(self.testing_dataloader):
            image = image.to(self.device)
            output = self.model(image)
            output_label = torch.argmax(output, dim=1)
            outputs[i] = output_label.cpu()[0]
        return outputs

    def make_inference(self):
        outputs = self.inference()
        fhand = open(self.output_file, "w")
        fhand.write("ImageId,Label\n")
        for key, value in outputs.items():
            fhand.write(f"{key + 1},{value}")
            fhand.write("\n")
        fhand.close()

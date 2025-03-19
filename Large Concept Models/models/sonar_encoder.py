import torch
import torch.nn as nn
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

class SonarEncoder(nn.Module):
    def __init__(self, encoder_name="text_sonar_basic_encoder", device=torch.device("cpu"), dtype=torch.float32):
        super().__init__()
        self.t2vec_model = TextToEmbeddingModelPipeline(
            encoder=encoder_name,
            tokenizer=encoder_name,
            device=device,
            dtype=dtype,
        )

    def forward(self, text_batch, source_lang="eng_Latn"):
        return self.t2vec_model.predict(text_batch, source_lang=source_lang, batch_size=len(text_batch), target_device=self.t2vec_model.device)
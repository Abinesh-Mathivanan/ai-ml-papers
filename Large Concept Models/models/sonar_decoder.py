import torch
import torch.nn as nn
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

class SonarDecoder(nn.Module):
    def __init__(self, decoder_name="text_sonar_basic_decoder", device=torch.device("cpu"), dtype=torch.float32):
        super().__init__()
        self.vec2text_model = EmbeddingToTextModelPipeline(
            decoder=decoder_name,
            tokenizer=decoder_name,
            device=device,
            dtype=dtype,
        )

    def forward(self, concept_embeddings, target_lang="eng_Latn"):
        return self.vec2text_model.predict(concept_embeddings, target_lang=target_lang, batch_size=concept_embeddings.size(0))
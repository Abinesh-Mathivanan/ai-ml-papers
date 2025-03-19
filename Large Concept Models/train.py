import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from models import SonarEncoder, SonarDecoder, LargeConceptModel, RobustScaler
from data.preprocess import load_text_data

def train():
    input_dim = 1024
    model_dim = 512
    output_dim = 1024
    num_layers = 6
    num_heads = 8
    batch_size = 2
    context_seq_len = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_data = load_text_data("data")
    scaler = RobustScaler()
    dummy_sonar_data = torch.randn(10000, input_dim)
    scaler.fit(dummy_sonar_data)

    model = LargeConceptModel(input_dim, model_dim, output_dim, num_layers, num_heads, scaler).to(device)
    sonar_encoder = SonarEncoder(device="cpu", dtype=torch.float32)
    sonar_decoder = SonarDecoder(device="cpu", dtype=torch.float32)

    text_batch = text_data[:batch_size]
    target_text_batch = text_data[batch_size:batch_size * 2]

    input_concept_embeddings = sonar_encoder(text_batch, source_lang="eng_Latn").unsqueeze(1).to(device)
    input_concept_embeddings = input_concept_embeddings.clone().detach().requires_grad_(True)

    target_concept_embeddings = sonar_encoder(target_text_batch, source_lang="eng_Latn")
    target_concept_embeddings = target_concept_embeddings.clone().detach().requires_grad_(True).to(device)
    target_concept_embeddings = target_concept_embeddings.reshape(batch_size, 1, output_dim)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    epochs = 2
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(input_concept_embeddings, memory=input_concept_embeddings[:, :context_seq_len, :])
        loss = criterion(outputs, target_concept_embeddings)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()

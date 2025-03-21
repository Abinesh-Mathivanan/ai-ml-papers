import torch
from models import SonarEncoder, SonarDecoder, LargeConceptModel, RobustScaler

# We used SONAR to implement Sentence level embedding (reference - https://ai.meta.com/research/publications/sonar-sentence-level-multimodal-and-language-agnostic-representations/)
# it's quite a heavy package, i suggest to test the model in colab env. 

def main():
    input_dim = 512
    model_dim = 768
    output_dim = 512
    num_layers = 6
    num_heads = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # better use T4-V2 GPU 
    scaler = RobustScaler()

    encoder = SonarEncoder(device=device)
    decoder = SonarDecoder(device=device)
    model = LargeConceptModel(input_dim, model_dim, output_dim, num_layers, num_heads, scaler).to(device)

    print("Models initialized successfully.")
    
    input_texts = ["This is a sample input for the Sonar model."]
    encoded_embeddings = encoder(input_texts).to(device)
    model_output = model(encoded_embeddings)
    decoded_texts = decoder(model_output)
    
    print("\nDecoded Output:")
    print(decoded_texts[0])

if __name__ == "__main__":
    main()

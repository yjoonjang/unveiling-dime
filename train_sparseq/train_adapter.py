import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class QueryAdapter(nn.Module):
    def __init__(self, embedding_dim, hidden_dim_ratio=0.5):
        super().__init__()
        hidden_dim = int(embedding_dim * hidden_dim_ratio)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)

class PairedDocDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset = load_from_disk(dataset_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item['anchor'], item['positive']

def collate_fn(batch):
    queries, docs = zip(*batch)
    return list(queries), list(docs)

def train(model_name: str, output_dir: str, epochs: int, batch_size: int, lr: float, device: str, dataset_path: str, query_prompt: str, doc_prompt: str, log_steps: int):
    
    os.makedirs(output_dir, exist_ok=True)
    model_safe_name = model_name.replace('/', '_')
    output_path = os.path.join(output_dir, f'{model_safe_name}_adapter.pt')

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()

    adapter = QueryAdapter(embedding_dim).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    print(f"Loading dataset from: {dataset_path}")
    dataset = PairedDocDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    model.eval()
    adapter.train()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for i, (queries, docs) in enumerate(progress_bar):
            processed_queries = [query_prompt + q for q in queries] if query_prompt else queries
            processed_docs = [doc_prompt + d for d in docs] if doc_prompt else docs
            
            with torch.no_grad():
                query_embeddings_inf = model.encode(processed_queries, convert_to_tensor=True, device=device)
                doc_embeddings = model.encode(processed_docs, convert_to_tensor=True, device=device)
            
            # Clone the inference tensor to use it in the autograd graph
            query_embeddings = query_embeddings_inf.clone()
            
            adapter_output = adapter(query_embeddings)
            predicted_interaction = query_embeddings * adapter_output
            target_interaction = query_embeddings_inf * doc_embeddings

            loss = loss_fn(predicted_interaction, target_interaction)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (i + 1)})

            if (i + 1) % log_steps == 0:
                print(f"  [Step {i+1}/{len(dataloader)}] Current Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.6f}")

        print(f"Saving adapter to {output_path}")
        torch.save(adapter.state_dict(), output_path)

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Sparseq Query Adapter.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="trained_adapters")
    parser.add_argument("--dataset_path", type=str, default="/data_x/yjoonjang/SPARSEQ/DATA/miracl_train")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--query_prompt", type=str, default=None)
    parser.add_argument("--doc_prompt", type=str, default=None)
    parser.add_argument("--log_steps", type=int, default=10, help="Log training loss every N steps.")
    
    args = parser.parse_args()
    train(**vars(args))

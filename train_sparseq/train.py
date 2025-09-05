import argparse
import os
import torch
import ir_datasets
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    model_safe_name = model_name.replace('/', '_')
    output_path = os.path.join(output_dir, f'{model_safe_name}_P.pt')

    # Load sentence transformer model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()

    # Initialize parameter P
    P = torch.nn.Parameter(torch.randn(embedding_dim, device=device))

    # Setup optimizer and loss function
    optimizer = torch.optim.AdamW([P], lr=lr)
    loss_fn = torch.nn.MSELoss()

    # Load dataset and dataloader
    print(f"Loading dataset from: {dataset_path}")
    dataset = PairedDocDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    model.eval()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for i, (queries, docs) in enumerate(progress_bar):
            
            processed_queries = [query_prompt + q for q in queries] if query_prompt else queries
            processed_docs = [doc_prompt + d for d in docs] if doc_prompt else docs
            
            with torch.no_grad():
                query_embeddings = torch.tensor(model.encode(processed_queries, convert_to_tensor=False), device=device)
                doc_embeddings = torch.tensor(model.encode(processed_docs, convert_to_tensor=False), device=device)
            
            # Compute interactions
            target_interaction = query_embeddings * doc_embeddings
            predicted_interaction = query_embeddings * P

            # Compute loss
            loss = loss_fn(predicted_interaction, target_interaction)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (i + 1)})

            if (i + 1) % log_steps == 0:
                print(f"  [Step {i+1}/{len(dataloader)}] Current Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.6f}")

        # Save parameter P after each epoch
        print(f"Saving parameter P to {output_path}")
        torch.save(P, output_path)

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sparseq parameter P.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the SentenceTransformer model to use.")
    parser.add_argument("--output_dir", type=str, default="./trained_params", help="Directory to save the trained parameter P.")
    parser.add_argument("--dataset_path", type=str, default="/data_x/yjoonjang/SPARSEQ/DATA/miracl_train", help="Path to the training dataset.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to train on (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--query_prompt", type=str, default=None, help="Prompt to prepend to queries.")
    parser.add_argument("--doc_prompt", type=str, default=None, help="Prompt to prepend to documents.")
    parser.add_argument("--log_steps", type=int, default=10, help="Log training loss every N steps.")
    
    args = parser.parse_args()
    
    train(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        dataset_path=args.dataset_path,
        query_prompt=args.query_prompt,
        doc_prompt=args.doc_prompt,
        log_steps=args.log_steps
    )

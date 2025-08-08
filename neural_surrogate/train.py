import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import MLP
from dataset import BurgersDataset
import argparse

def train(args):
    dataset = BurgersDataset(data_dir=args.data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MLP(input_size=128, hidden_size=args.hidden_size, output_size=128)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                y_pred = model(x)
                val_loss += criterion(y_pred, y).item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural surrogate for the 1D Burgers equation.')
    parser.add_argument('--data_dir', type=str, default='/home/dan/Desktop/neural-adapter/precice_datagen/datagen', help='Directory containing the training data.')
    parser.add_argument('--model_path', type=str, default='burgers_surrogate.pth', help='Path to save the trained model.')
    parser.add_argument('--hidden_size', type=int, default=256, help='Number of neurons in hidden layers.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    
    args = parser.parse_args()
    train(args)
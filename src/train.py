import torch
import torch.optim as optim
from model import NeRFModel
from dataloader import create_nerf_dataloader
from utils import compute_loss  # Assuming the compute_loss function is in a separate file

def train_nerf(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create DataLoader
    dataloader = create_nerf_dataloader(
        config.dataset_path,
        (config.image_height, config.image_width),
        config.camera_matrix,
        config.num_sample_points,
        config.batch_size
    )

    # Initialize model
    model = NeRFModel().to(device)

    # Initialize optimizer
    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    # Train model
    model.train()
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            img = batch['image'].to(device)
            sample_points = batch['sample_points'].to(device)
            view_directions = torch.randn_like(sample_points)  # Use random view directions for this example

            alpha, rgb = model(sample_points, view_directions)
            loss = compute_loss(alpha, rgb, img, sample_points, config)  # Assuming the compute_loss function is in a separate file

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {epoch_loss / len(dataloader)}")

    # Save the trained model
    torch.save(model.state_dict(), f"{config.model_checkpoint_dir}/model.pth")

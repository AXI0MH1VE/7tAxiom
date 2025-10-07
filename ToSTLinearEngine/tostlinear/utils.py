import torch

def create_sample_data(input_size, batch_size=1):
    return torch.randn(batch_size, input_size)

def calculate_loss(predictions, targets, loss_fn=nn.MSELoss()):
    return loss_fn(predictions, targets)

def train_model(model, data, targets, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(data)
        loss = calculate_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

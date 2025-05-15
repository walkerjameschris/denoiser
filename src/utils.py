import torch

def scale_images(x, apply=True, mean=115, std=75):
    
    if apply:
        return (x - mean) / std
    else:
        return (x * std) + mean

def add_noise(x, scale=25.0, low=0, high=255):

    # Generate noise)
    noise = torch.rand(x.shape)
    noise -= scale * 2
    noise += scale

    # Apply noise and cap/floor
    return torch.clip(x + noise, low, high)

def get_batch(images, n=1000):

    # Get batch of random image IDs
    n_image, _, width, height = images.shape
    image_ids = torch.randperm(n_image)[:min(n_image, n)]

    # Select images and add noise
    batch = images[image_ids, :, :]
    batch_noise = add_noise(images[image_ids, :, :])

    # Scale images
    batch = scale_images(batch)
    batch_noise = scale_images(batch_noise)

    return batch, batch_noise

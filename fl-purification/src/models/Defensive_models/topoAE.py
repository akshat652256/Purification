from topoAE.scripts.mnist_gen import load_trained_model, infer_with_loaded_model

model = load_trained_model("path/to/model.pth", device='cuda')
"""
        image_array = np.array(image) / 255.0
Note: we are dividing by 255 here change if needed.
"""

# # Load model once
# model = load_trained_model("path/to/model.pth", device='cuda')

# # With PIL Image
# from PIL import Image
# img = Image.open("path/to/image.png")
# latent, reconstructed = infer_with_loaded_model(model, img, device='cuda')

# # With numpy array
# img_array = np.random.rand(28, 28)  # Your image data
# latent, reconstructed = infer_with_loaded_model(model, img_array, device='cuda')

# # Process many images efficiently
# for image_path in image_list:
#     latent, reconstructed = infer_with_loaded_model(model, image_path, device='cuda')

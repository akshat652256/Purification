from topoAE.scripts.mnist_gen import load_trained_model, infer_with_loaded_model

model = load_trained_model("path/to/model.pth", device='cuda')

# # Use the model multiple times without reloading
# latent1, reconstructed1 = infer_with_loaded_model(model, "image1.png", device='cuda')
# latent2, reconstructed2 = infer_with_loaded_model(model, "image2.png", device='cuda')
# latent3, reconstructed3 = infer_with_loaded_model(model, image_array, device='cuda')

# # Process many images efficiently
# for image_path in image_list:
#     latent, reconstructed = infer_with_loaded_model(model, image_path, device='cuda')

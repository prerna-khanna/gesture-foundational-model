import torch
import os

# Path to your .pt file
model_path = "unihar_impl/unihar_bert_fed_d1.pt"

# Load the model
try:
    # Try loading the entire model (architecture + weights)
    model = torch.load(model_path, map_location=torch.device('cpu'))
    print("Successfully loaded full model.")
except Exception as e:
    print(f"Error loading full model: {e}")
    try:
        # Try loading as state dict (just weights)
        model = torch.load(model_path, map_location=torch.device('cpu'))
        print("Loaded as state dictionary.")
    except Exception as e2:
        print(f"Error loading state dict: {e2}")

# Print model information
print("\n=== Model Information ===")

# If it's a model object
if hasattr(model, 'eval'):
    model.eval()  # Set to evaluation mode
    print(model)  # Print architecture
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print shape of first few parameters
    print("\nSample parameter shapes:")
    for name, param in list(model.named_parameters())[:5]:
        print(f"{name}: {param.shape}")

# If it's a state dictionary
elif isinstance(model, dict) and 'state_dict' in model:
    state_dict = model['state_dict']
    print("Model contains state dictionary with keys:")
    for key in state_dict.keys():
        print(f"  - {key}: {state_dict[key].shape}")
elif isinstance(model, dict):
    print("Model is a dictionary with keys:")
    for key in model.keys():
        if isinstance(model[key], torch.Tensor):
            print(f"  - {key}: {model[key].shape}")
        else:
            print(f"  - {key}: {type(model[key])}")

# Try to identify input size if possible
print("\nTrying to identify input dimensions...")
try:
    # Look for the first layer's weight shape
    first_layer_params = None
    if hasattr(model, 'parameters'):
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                first_layer_params = param
                break
    
    if first_layer_params is not None:
        input_size = first_layer_params.shape[1]
        print(f"Possible input size: {input_size}")
    else:
        print("Could not determine input size from model parameters.")
except Exception as e:
    print(f"Error determining input size: {e}")
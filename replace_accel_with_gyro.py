import numpy as np

def transform_array(input_array):
    """
    Transform an array of shape (n,120,6) by:
    1. Removing the first 3 elements from the last dimension
    2. Repeating the last 3 elements to maintain shape (n,120,6)
    
    Args:
        input_array: NumPy array of shape (n,120,6)
        
    Returns:
        Transformed array of shape (n,120,6)
    """
    # Verify input shape
    if input_array.shape[-1] != 6:
        raise ValueError(f"Last dimension must be 6, got {input_array.shape[-1]}")
    
    # Get the last 3 elements from the last dimension
    last_three = input_array[..., 3:6]
    
    # Repeat the last 3 elements to create a new array of shape (n,120,6)
    # By concatenating the last 3 elements with themselves
    result = np.concatenate([last_three, last_three], axis=-1)
    
    return result

# Example usage
if __name__ == "__main__":
    # read npy file
    sample = np.load("dataset/blind_user_filtered/data_20_120.npy")

    
    # Apply the transformation
    transformed = transform_array(sample)
    
    # Verify the shape remains the same
    print(f"Original shape: {sample.shape}")
    print(f"Transformed shape: {transformed.shape}")
    
    # Print a small slice to verify the transformation worked correctly
    print("\nOriginal slice (first row, first column):")
    print(sample[0, 0, :])
    print("\nTransformed slice (first row, first column):")
    print(transformed[0, 0, :])

    # Save the transformed array
    np.save("dataset/blind_user_filtered/data_20_120_transformed.npy", transformed)
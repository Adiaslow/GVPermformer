# test_imports.py
import sys
import torch

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

try:
    import torch_geometric

    print(f"PyTorch Geometric version: {torch_geometric.__version__}")

    # Test importing specific modules
    from torch_geometric.data import Data

    print("Successfully imported Data")

    from torch_geometric.data import Batch

    print("Successfully imported Batch")

    # Try loading a dataset
    test_data_path = "data/processed/test_data.pt"
    print(f"Attempting to load: {test_data_path}")

    # Add safe globals
    torch.serialization.add_safe_globals([Data])

    # Try loading with weights_only=False
    try:
        data = torch.load(test_data_path, weights_only=False)
        print(f"Successfully loaded data with weights_only=False: {type(data)}")
        print(f"Dataset length: {len(data)}")
    except Exception as e:
        print(f"Error loading with weights_only=False: {str(e)}")

    # Try loading with map_location
    try:
        data = torch.load(test_data_path, map_location="cpu", weights_only=False)
        print(f"Successfully loaded data with map_location: {type(data)}")
    except Exception as e:
        print(f"Error loading with map_location: {str(e)}")

except ImportError as e:
    print(f"Error importing torch_geometric: {str(e)}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")

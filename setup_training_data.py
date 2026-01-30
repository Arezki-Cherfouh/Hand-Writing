import os

def setup_training_folders():
    """Create the training_data folder structure"""
    
    base_dir = "training_data"
    
    print("Setting up training data folders...")
    print(f"\nCreating directory: {base_dir}/")
    
    # Create base directory
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"  ✓ Created {base_dir}/")
    else:
        print(f"  ✓ {base_dir}/ already exists")
    
    # Create digit folders (0-9)
    for digit in range(10):
        digit_dir = os.path.join(base_dir, str(digit))
        if not os.path.exists(digit_dir):
            os.makedirs(digit_dir)
            print(f"  ✓ Created {base_dir}/{digit}/")
        else:
            print(f"  ✓ {base_dir}/{digit}/ already exists")
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60)
    print("\nFolder structure created:")
    print(f"  {base_dir}/")
    print("    ├── 0/  (place images of digit 0 here)")
    print("    ├── 1/  (place images of digit 1 here)")
    print("    ├── 2/  (place images of digit 2 here)")
    print("    ├── 3/  (place images of digit 3 here)")
    print("    ├── 4/  (place images of digit 4 here)")
    print("    ├── 5/  (place images of digit 5 here)")
    print("    ├── 6/  (place images of digit 6 here)")
    print("    ├── 7/  (place images of digit 7 here)")
    print("    ├── 8/  (place images of digit 8 here)")
    print("    └── 9/  (place images of digit 9 here)")

if __name__ == "__main__":
    setup_training_folders()
import os

def setup_character_folders(custom_classes=None):
    base_dir = "character_data"
    
    print("="*60)
    print("Character Data Folder Setup")
    print("="*60)
    
    # Create base directory
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"\n✓ Created {base_dir}/")
    else:
        print(f"\n✓ {base_dir}/ already exists")
    
    # Get class names from user if not provided
    if custom_classes is None:
        print("\nWhat type of characters do you want to train on?")
        print("1. Uppercase letters (A-Z)")
        print("2. Lowercase letters (a-z)")
        print("3. Digits (0-9)")
        print("4. Custom classes (you specify)")
        
        choice = input("\nSelect option (1-4) [default: 3]: ").strip()
        if choice == "" or choice == "3":
            custom_classes = [str(i) for i in range(10)]
        elif choice == "1":
            custom_classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        elif choice == "2":
            custom_classes = [chr(i) for i in range(ord('a'), ord('z')+1)]
        elif choice == "4":
            print("\nEnter class names separated by commas (e.g., A,B,C or cat,dog,bird)")
            classes_input = input("Classes: ").strip()
            custom_classes = [c.strip() for c in classes_input.split(',') if c.strip()]
            if not custom_classes:
                print("No classes entered. Using digits 0-9 as default.")
                custom_classes = [str(i) for i in range(10)]
        else:
            print("Invalid choice. Using digits 0-9 as default.")
            custom_classes = [str(i) for i in range(10)]
    
    # Create class folders
    print(f"\nCreating folders for {len(custom_classes)} classes...")
    for class_name in custom_classes:
        # Sanitize folder name (remove invalid characters)
        safe_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_name:
            safe_name = f"class_{custom_classes.index(class_name)}"
        
        class_dir = os.path.join(base_dir, safe_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            print(f"  ✓ Created {base_dir}/{safe_name}/")
        else:
            print(f"  ✓ {base_dir}/{safe_name}/ already exists")
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60)
    
    print(f"\nFolder structure created in '{base_dir}/':")
    print(f"Number of classes: {len(custom_classes)}")
    if len(custom_classes) <= 20:
        print(f"Classes: {', '.join(custom_classes)}")
    else:
        print(f"Classes: {', '.join(custom_classes[:10])} ... (and {len(custom_classes)-10} more)")
    
    print("\nExample folder structure:")
    print(f"  {base_dir}/")
    for i, class_name in enumerate(custom_classes[:5]):
        safe_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_name:
            safe_name = f"class_{i}"
        print(f"    ├── {safe_name}/")
        print(f"    │   ├── image1.png")
        print(f"    │   ├── image2.jpg")
        print(f"    │   └── ...")
    if len(custom_classes) > 5:
        print(f"    └── ... ({len(custom_classes)-5} more folders)")

if __name__ == "__main__":
    setup_character_folders()
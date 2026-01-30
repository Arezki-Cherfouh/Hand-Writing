import sys
import os
import pygame
import numpy as np
import json
from tkinter import Tk, filedialog

# Check for TensorFlow/Keras availability
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("Error: TensorFlow is not installed.")
    print("Please install it with: pip install tensorflow")
    sys.exit(1)

# Check for PIL availability
try:
    from PIL import Image
except ImportError:
    print("Error: PIL (Pillow) is not installed.")
    print("Please install it with: pip install pillow")
    sys.exit(1)

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 32
CELL_SIZE = 16
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 50

# Colors (Dark style)
BACKGROUND = (30, 30, 30)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRID_BG = (45, 45, 45)
GRID_LINE = (70, 70, 70)
TEXT_PRIMARY = (255, 255, 255)
TEXT_SECONDARY = (255, 255, 255)
BLUE = (66, 135, 245)
RED = (220, 53, 69)
GREEN = (40, 200, 80)
PURPLE = (156, 39, 176)
ORANGE = (255, 152, 0)

# Button dimensions
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 50
BUTTON_MARGIN = 20


def find_available_models():
    """Find all .h5 model files in the current directory"""
    models = [f for f in os.listdir('.') if f.endswith('.h5')]
    return models


def load_class_mapping(model_filename):
    """Load class mapping JSON file if it exists"""
    # Try to find matching class mapping file
    base_name = model_filename.replace('.h5', '')
    mapping_file = base_name.replace('character_model_', 'class_mapping_') + '.json'
    
    if not mapping_file.startswith('class_mapping_'):
        mapping_file = 'class_mapping_' + base_name + '.json'
    
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
            # Convert string keys to int keys
            return {int(k): v for k, v in mapping.items()}
        except Exception as e:
            print(f"Warning: Could not load class mapping: {e}")
    
    return None


def select_model():
    """Let user select which model to load"""
    models = find_available_models()
    
    if not models:
        print("Error: No model files (.h5) found in current directory.")
        print("Please train a model first using train_universal.py or train.py")
        return None, None
    
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    if len(models) == 1:
        print(f"\nAutomatically selecting: {models[0]}")
        selected = models[0]
    else:
        while True:
            try:
                choice = input(f"\nSelect model (1-{len(models)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    selected = models[idx]
                    break
                print(f"Invalid choice. Please enter 1-{len(models)}.")
            except (ValueError, KeyboardInterrupt):
                print("\nNo model selected. Exiting.")
                return None, None
    
    # Load the model
    try:
        print(f"Loading model: {selected}")
        model = keras.models.load_model(selected)
        print("Model loaded successfully!")
        
        # Load class mapping
        class_mapping = load_class_mapping(selected)
        if class_mapping:
            print(f"Class mapping loaded: {len(class_mapping)} classes")
            print(f"Classes: {', '.join([str(v) for v in class_mapping.values()])}")
        else:
            # Create default numeric mapping
            output_shape = model.output_shape[-1]
            class_mapping = {i: str(i) for i in range(output_shape)}
            print(f"Using default numeric mapping: 0-{output_shape-1}")
        
        return model, class_mapping
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


# Load model at startup
print("="*60)
print("Universal Character Recognition")
print("="*60)
model, class_mapping = select_model()

if model is None:
    sys.exit(1)

num_classes = len(class_mapping)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Universal Character Recognition")

# Fonts
title_font = pygame.font.Font(None, 48)
button_font = pygame.font.Font(None, 32)
result_font = pygame.font.Font(None, 64)
info_font = pygame.font.Font(None, 24)
small_font = pygame.font.Font(None, 20)

# Create the drawing grid (32x32)
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

# Prediction result
prediction_idx = None
prediction_label = None
confidence = None
loaded_image_path = None


# Button class
class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover = False
    
    def draw(self, surface):
        color = self.color if not self.hover else tuple(min(c + 30, 255) for c in self.color)
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, tuple(min(c + 40, 255) for c in self.color), self.rect, 2, border_radius=8)
        
        text_surface = button_font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
    
    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)
    
    def update_hover(self, pos):
        self.hover = self.rect.collidepoint(pos)


# Create buttons
classify_button = Button(
    GRID_OFFSET_X + GRID_SIZE * CELL_SIZE + BUTTON_MARGIN,
    GRID_OFFSET_Y + 50,
    BUTTON_WIDTH,
    BUTTON_HEIGHT,
    "Classify",
    BLUE
)

reset_button = Button(
    GRID_OFFSET_X + GRID_SIZE * CELL_SIZE + BUTTON_MARGIN,
    GRID_OFFSET_Y + 50 + BUTTON_HEIGHT + BUTTON_MARGIN,
    BUTTON_WIDTH,
    BUTTON_HEIGHT,
    "Reset",
    RED
)

load_image_button = Button(
    GRID_OFFSET_X + GRID_SIZE * CELL_SIZE + BUTTON_MARGIN,
    GRID_OFFSET_Y + 50 + (BUTTON_HEIGHT + BUTTON_MARGIN) * 2,
    BUTTON_WIDTH,
    BUTTON_HEIGHT,
    "Load Image",
    PURPLE
)


def draw_grid():
    """Draw the 32x32 grid with dark theme"""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x = GRID_OFFSET_X + col * CELL_SIZE
            y = GRID_OFFSET_Y + row * CELL_SIZE
            
            # Draw cell with intensity based on grid value
            intensity = int(grid[row, col] * 255)
            r = min(255, max(0, GRID_BG[0] + intensity))
            g = min(255, max(0, GRID_BG[1] + intensity))
            b = min(255, max(0, GRID_BG[2] + intensity))
            color = (r, g, b)
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
            
            # Draw grid lines
            pygame.draw.rect(screen, GRID_LINE, (x, y, CELL_SIZE, CELL_SIZE), 1)


def paint_cell(pos, brush_size=2):
    """Paint cells when mouse is pressed"""
    mouse_x, mouse_y = pos
    
    if (GRID_OFFSET_X <= mouse_x < GRID_OFFSET_X + GRID_SIZE * CELL_SIZE and
        GRID_OFFSET_Y <= mouse_y < GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE):
        
        grid_x = (mouse_x - GRID_OFFSET_X) // CELL_SIZE
        grid_y = (mouse_y - GRID_OFFSET_Y) // CELL_SIZE
        
        for dy in range(-brush_size + 1, brush_size):
            for dx in range(-brush_size + 1, brush_size):
                new_x = grid_x + dx
                new_y = grid_y + dy
                
                if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                    distance = abs(dx) + abs(dy)
                    intensity = max(0, 1.0 - distance * 0.3)
                    grid[new_y, new_x] = min(1.0, grid[new_y, new_x] + intensity)


def load_image_to_grid():
    """Load an image file and convert it to the 32x32 grid"""
    global grid, loaded_image_path, prediction_idx, prediction_label, confidence
    
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    
    if file_path:
        try:
            img = Image.open(file_path)
            img_gray = img.convert('L')
            img_resized = img_gray.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            
            if np.mean(img_array) > 0.5:
                img_array = 1.0 - img_array
            
            grid[:] = img_array
            loaded_image_path = os.path.basename(file_path)
            prediction_idx = None
            prediction_label = None
            confidence = None
            
            print(f"Loaded image: {file_path}")
            
        except Exception as e:
            print(f"Error loading image: {e}")
            loaded_image_path = None


def classify_character():
    """Classify the drawn character"""
    global prediction_idx, prediction_label, confidence
    
    if np.max(grid) < 0.1:
        prediction_idx = None
        prediction_label = None
        confidence = None
        return
    
    from scipy import ndimage
    
    rows = np.any(grid > 0.1, axis=1)
    cols = np.any(grid > 0.1, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        prediction_idx = None
        prediction_label = None
        confidence = None
        return
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    digit = grid[rmin:rmax+1, cmin:cmax+1]
    height, width = digit.shape
    scale = min(20.0 / height, 20.0 / width)
    
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    digit_resized = ndimage.zoom(digit, (new_height / height, new_width / width))
    
    image_28x28 = np.zeros((28, 28), dtype=np.float32)
    y_offset = (28 - new_height) // 2
    x_offset = (28 - new_width) // 2
    
    image_28x28[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = digit_resized
    model_input = image_28x28.reshape(1, 28, 28, 1)
    
    predictions = model.predict(model_input, verbose=0)
    prediction_idx = np.argmax(predictions[0])
    prediction_label = class_mapping.get(prediction_idx, str(prediction_idx))
    confidence = predictions[0][prediction_idx]


def reset_grid():
    """Reset the drawing grid"""
    global grid, prediction_idx, prediction_label, confidence, loaded_image_path
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    prediction_idx = None
    prediction_label = None
    confidence = None
    loaded_image_path = None


# Main game loop
running = True
clock = pygame.time.Clock()
drawing = False

print("\nInstructions:")
print("- Draw a character on the grid by clicking and dragging")
print("- OR click 'Load Image' to load an image file")
print("- Click 'Classify' to recognize the character")
print("- Click 'Reset' to clear the grid")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                pos = pygame.mouse.get_pos()
                
                if classify_button.is_clicked(pos):
                    classify_character()
                elif reset_button.is_clicked(pos):
                    reset_grid()
                elif load_image_button.is_clicked(pos):
                    load_image_to_grid()
                else:
                    drawing = True
                    paint_cell(pos)
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
        
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                paint_cell(pygame.mouse.get_pos())
    
    # Update button hover states
    mouse_pos = pygame.mouse.get_pos()
    classify_button.update_hover(mouse_pos)
    reset_button.update_hover(mouse_pos)
    load_image_button.update_hover(mouse_pos)
    
    # Clear screen
    screen.fill(BACKGROUND)
    
    # Draw title
    title_surface = title_font.render("Character Recognition", True, TEXT_PRIMARY)
    screen.blit(title_surface, (WINDOW_WIDTH // 2 - title_surface.get_width() // 2, 10))
    
    # Draw instructions
    instruction_surface = info_font.render("Draw a character or load an image", True, TEXT_SECONDARY)
    screen.blit(instruction_surface, (GRID_OFFSET_X, GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE + 20))
    
    # Show loaded image name
    if loaded_image_path:
        loaded_text = small_font.render(f"Loaded: {loaded_image_path}", True, PURPLE)
        screen.blit(loaded_text, (GRID_OFFSET_X, GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE + 45))
    
    # Draw grid
    draw_grid()
    
    # Draw buttons
    classify_button.draw(screen)
    reset_button.draw(screen)
    load_image_button.draw(screen)
    
    # Draw prediction result
    if prediction_label is not None:
        result_y = GRID_OFFSET_Y + 250
        
        # Draw "Prediction:" label
        label_surface = button_font.render("Prediction:", True, TEXT_PRIMARY)
        screen.blit(label_surface, (
            classify_button.rect.x + BUTTON_WIDTH // 2 - label_surface.get_width() // 2,
            result_y
        ))
        
        # Draw the predicted character (handle long labels)
        label_text = str(prediction_label)
        if len(label_text) > 3:
            char_surface = button_font.render(label_text, True, GREEN)
        else:
            char_surface = result_font.render(label_text, True, GREEN)
        
        screen.blit(char_surface, (
            classify_button.rect.x + BUTTON_WIDTH // 2 - char_surface.get_width() // 2,
            result_y + 40
        ))
        
        # Draw confidence
        confidence_text = f"Confidence: {confidence*100:.1f}%"
        confidence_surface = info_font.render(confidence_text, True, TEXT_SECONDARY)
        screen.blit(confidence_surface, (
            classify_button.rect.x + BUTTON_WIDTH // 2 - confidence_surface.get_width() // 2,
            result_y + 110
        ))
    
    # Update display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
print("\nGoodbye!")
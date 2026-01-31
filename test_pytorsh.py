# import sys
# import os
# import pygame
# import numpy as np
# from tkinter import Tk, filedialog

# # Check for PyTorch availability
# try:
#     import torch
#     import torch.nn as nn
# except ImportError:
#     print("Error: PyTorch is not installed.")
#     print("Please install it with: pip install torch")
#     sys.exit(1)

# # Check for PIL availability
# try:
#     from PIL import Image
# except ImportError:
#     print("Error: PIL (Pillow) is not installed.")
#     print("Please install it with: pip install pillow")
#     sys.exit(1)

# # Constants
# WINDOW_WIDTH = 800
# WINDOW_HEIGHT = 600
# GRID_SIZE = 32
# CELL_SIZE = 16  # 32x32 grid with 16px cells = 512x512 drawing area
# GRID_OFFSET_X = 50
# GRID_OFFSET_Y = 50

# # Colors (Dark style)
# BACKGROUND = (30, 30, 30)
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# GRID_BG = (45, 45, 45)
# GRID_LINE = (70, 70, 70)
# TEXT_PRIMARY = (255, 255, 255)  # Pure white
# TEXT_SECONDARY = (255, 255, 255)  # Pure white
# BLUE = (66, 135, 245)
# RED = (220, 53, 69)
# GREEN = (40, 200, 80)
# PURPLE = (156, 39, 176)

# # Button dimensions
# BUTTON_WIDTH = 150
# BUTTON_HEIGHT = 50
# BUTTON_MARGIN = 20

# # Define the CNN model (same as in training)
# class HandwritingCNN(nn.Module):
#     def __init__(self):
#         super(HandwritingCNN, self).__init__()
        
#         # First convolutional layer - 32 filters, 3x3 kernel
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Second convolutional layer - 64 filters, 3x3 kernel
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Fully connected layers
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(64 * 5 * 5, 128)
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(128, 10)
        
#         # Activation functions
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         # First conv block
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool1(x)
        
#         # Second conv block
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.pool2(x)
        
#         # Flatten and fully connected layers
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
        
#         return x

# # Load the trained model
# print("Loading trained model...")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = HandwritingCNN().to(device)

# try:
#     checkpoint = torch.load("handwriting_model.pt", map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()  # Set to evaluation mode
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     print("Please train the model first by running: python train_pytorch.py")
#     sys.exit(1)

# # Initialize Pygame
# pygame.init()
# screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
# pygame.display.set_caption("Handwriting Recognition - PyTorch - Dark Mode + Image Loading")

# # Fonts
# title_font = pygame.font.Font(None, 48)
# button_font = pygame.font.Font(None, 32)
# result_font = pygame.font.Font(None, 64)
# info_font = pygame.font.Font(None, 24)
# small_font = pygame.font.Font(None, 20)

# # Create the drawing grid (32x32)
# grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

# # Prediction result
# prediction = None
# confidence = None
# loaded_image_path = None

# # Button class
# class Button:
#     def __init__(self, x, y, width, height, text, color):
#         self.rect = pygame.Rect(x, y, width, height)
#         self.text = text
#         self.color = color
#         self.hover = False
    
#     def draw(self, surface):
#         color = self.color if not self.hover else tuple(min(c + 30, 255) for c in self.color)
#         pygame.draw.rect(surface, color, self.rect, border_radius=8)
#         pygame.draw.rect(surface, tuple(min(c + 40, 255) for c in self.color), self.rect, 2, border_radius=8)
        
#         text_surface = button_font.render(self.text, True, WHITE)
#         text_rect = text_surface.get_rect(center=self.rect.center)
#         surface.blit(text_surface, text_rect)
    
#     def is_clicked(self, pos):
#         return self.rect.collidepoint(pos)
    
#     def update_hover(self, pos):
#         self.hover = self.rect.collidepoint(pos)

# # Create buttons
# classify_button = Button(
#     GRID_OFFSET_X + GRID_SIZE * CELL_SIZE + BUTTON_MARGIN,
#     GRID_OFFSET_Y + 50,
#     BUTTON_WIDTH,
#     BUTTON_HEIGHT,
#     "Classify",
#     BLUE
# )

# reset_button = Button(
#     GRID_OFFSET_X + GRID_SIZE * CELL_SIZE + BUTTON_MARGIN,
#     GRID_OFFSET_Y + 50 + BUTTON_HEIGHT + BUTTON_MARGIN,
#     BUTTON_WIDTH,
#     BUTTON_HEIGHT,
#     "Reset",
#     RED
# )

# load_image_button = Button(
#     GRID_OFFSET_X + GRID_SIZE * CELL_SIZE + BUTTON_MARGIN,
#     GRID_OFFSET_Y + 50 + (BUTTON_HEIGHT + BUTTON_MARGIN) * 2,
#     BUTTON_WIDTH,
#     BUTTON_HEIGHT,
#     "Load Image",
#     PURPLE
# )

# def draw_grid():
#     """Draw the 32x32 grid with dark theme"""
#     for row in range(GRID_SIZE):
#         for col in range(GRID_SIZE):
#             x = GRID_OFFSET_X + col * CELL_SIZE
#             y = GRID_OFFSET_Y + row * CELL_SIZE
            
#             # Draw cell with intensity based on grid value (white drawing on dark background)
#             intensity = int(grid[row, col] * 255)
#             # Clamp values to ensure they're within 0-255 range
#             r = min(255, max(0, GRID_BG[0] + intensity))
#             g = min(255, max(0, GRID_BG[1] + intensity))
#             b = min(255, max(0, GRID_BG[2] + intensity))
#             color = (r, g, b)
#             pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
            
#             # Draw grid lines
#             pygame.draw.rect(screen, GRID_LINE, (x, y, CELL_SIZE, CELL_SIZE), 1)

# def paint_cell(pos, brush_size=2):
#     """Paint cells when mouse is pressed"""
#     mouse_x, mouse_y = pos
    
#     # Check if click is within grid
#     if (GRID_OFFSET_X <= mouse_x < GRID_OFFSET_X + GRID_SIZE * CELL_SIZE and
#         GRID_OFFSET_Y <= mouse_y < GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE):
        
#         # Calculate grid position
#         grid_x = (mouse_x - GRID_OFFSET_X) // CELL_SIZE
#         grid_y = (mouse_y - GRID_OFFSET_Y) // CELL_SIZE
        
#         # Paint with brush (paint surrounding cells too for smoother drawing)
#         for dy in range(-brush_size + 1, brush_size):
#             for dx in range(-brush_size + 1, brush_size):
#                 new_x = grid_x + dx
#                 new_y = grid_y + dy
                
#                 if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
#                     # Softer brush effect
#                     distance = abs(dx) + abs(dy)
#                     intensity = max(0, 1.0 - distance * 0.3)
#                     grid[new_y, new_x] = min(1.0, grid[new_y, new_x] + intensity)

# def load_image_to_grid():
#     """Load an image file and convert it to the 32x32 grid"""
#     global grid, loaded_image_path, prediction, confidence
    
#     # Hide the pygame window temporarily
#     root = Tk()
#     root.withdraw()
#     root.attributes('-topmost', True)
    
#     # Open file dialog
#     file_path = filedialog.askopenfilename(
#         title="Select an image file",
#         filetypes=[
#             ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
#             ("All files", "*.*")
#         ]
#     )
    
#     root.destroy()
    
#     if file_path:
#         try:
#             # Load the image
#             img = Image.open(file_path)
            
#             # Convert to grayscale
#             img_gray = img.convert('L')
            
#             # Resize to 32x32
#             img_resized = img_gray.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.LANCZOS)
            
#             # Convert to numpy array and normalize
#             img_array = np.array(img_resized).astype(np.float32) / 255.0
            
#             # Invert if needed (assuming white background with black digit)
#             # Check if the image has more white than black (average > 0.5)
#             if np.mean(img_array) > 0.5:
#                 img_array = 1.0 - img_array
            
#             # Update grid
#             grid[:] = img_array
#             loaded_image_path = os.path.basename(file_path)
            
#             # Reset prediction
#             prediction = None
#             confidence = None
            
#             print(f"Loaded image: {file_path}")
            
#         except Exception as e:
#             print(f"Error loading image: {e}")
#             loaded_image_path = None

# def classify_digit():
#     """Classify the drawn digit using PyTorch model"""
#     global prediction, confidence
    
#     # Check if grid has any content
#     if np.max(grid) < 0.1:
#         prediction = None
#         confidence = None
#         return
    
#     # Resize 32x32 to 28x28 for the model
#     from scipy import ndimage
    
#     # Find bounding box and center it
#     rows = np.any(grid > 0.1, axis=1)
#     cols = np.any(grid > 0.1, axis=0)
    
#     if not np.any(rows) or not np.any(cols):
#         prediction = None
#         confidence = None
#         return
    
#     rmin, rmax = np.where(rows)[0][[0, -1]]
#     cmin, cmax = np.where(cols)[0][[0, -1]]
    
#     # Extract the digit
#     digit = grid[rmin:rmax+1, cmin:cmax+1]
    
#     # Resize to fit in 20x20 box (leaving 4px border like MNIST)
#     height, width = digit.shape
#     scale = min(20.0 / height, 20.0 / width)
    
#     new_height = int(height * scale)
#     new_width = int(width * scale)
    
#     # Resize the digit
#     digit_resized = ndimage.zoom(digit, (new_height / height, new_width / width))
    
#     # Create 28x28 image with centered digit
#     image_28x28 = np.zeros((28, 28), dtype=np.float32)
    
#     # Center it
#     y_offset = (28 - new_height) // 2
#     x_offset = (28 - new_width) // 2
    
#     image_28x28[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = digit_resized
    
#     # Reshape for PyTorch model input (batch_size, channels, height, width)
#     model_input = torch.from_numpy(image_28x28).unsqueeze(0).unsqueeze(0).to(device)
    
#     # Make prediction
#     with torch.no_grad():
#         output = model(model_input)
#         probabilities = torch.nn.functional.softmax(output, dim=1)
#         prediction_tensor = torch.argmax(probabilities, dim=1)
#         confidence_tensor = probabilities[0][prediction_tensor]
        
#         prediction = prediction_tensor.item()
#         confidence = confidence_tensor.item()

# def reset_grid():
#     """Reset the drawing grid"""
#     global grid, prediction, confidence, loaded_image_path
#     grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
#     prediction = None
#     confidence = None
#     loaded_image_path = None

# # Main game loop
# running = True
# clock = pygame.time.Clock()
# drawing = False

# print("\nInstructions:")
# print("- Draw a digit (0-9) on the grid by clicking and dragging")
# print("- OR click 'Load Image' to load an image file")
# print("- Click 'Classify' to recognize the digit")
# print("- Click 'Reset' to clear the grid")

# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
        
#         elif event.type == pygame.MOUSEBUTTONDOWN:
#             if event.button == 1:  # Left click
#                 pos = pygame.mouse.get_pos()
                
#                 # Check button clicks
#                 if classify_button.is_clicked(pos):
#                     classify_digit()
#                 elif reset_button.is_clicked(pos):
#                     reset_grid()
#                 elif load_image_button.is_clicked(pos):
#                     load_image_to_grid()
#                 else:
#                     drawing = True
#                     paint_cell(pos)
        
#         elif event.type == pygame.MOUSEBUTTONUP:
#             if event.button == 1:
#                 drawing = False
        
#         elif event.type == pygame.MOUSEMOTION:
#             if drawing:
#                 paint_cell(pygame.mouse.get_pos())
    
#     # Update button hover states
#     mouse_pos = pygame.mouse.get_pos()
#     classify_button.update_hover(mouse_pos)
#     reset_button.update_hover(mouse_pos)
#     load_image_button.update_hover(mouse_pos)
    
#     # Clear screen with dark background
#     screen.fill(BACKGROUND)
    
#     # Draw title
#     title_surface = title_font.render("Handwriting Recognition (PyTorch)", True, TEXT_PRIMARY)
#     screen.blit(title_surface, (WINDOW_WIDTH // 2 - title_surface.get_width() // 2, 10))
    
#     # Draw instructions
#     instruction_surface = info_font.render("Draw a digit or load an image", True, TEXT_SECONDARY)
#     screen.blit(instruction_surface, (GRID_OFFSET_X, GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE + 20))
    
#     # Show loaded image name if applicable
#     if loaded_image_path:
#         loaded_text = small_font.render(f"Loaded: {loaded_image_path}", True, PURPLE)
#         screen.blit(loaded_text, (GRID_OFFSET_X, GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE + 45))
    
#     # Draw grid
#     draw_grid()
    
#     # Draw buttons
#     classify_button.draw(screen)
#     reset_button.draw(screen)
#     load_image_button.draw(screen)
    
#     # Draw prediction result
#     if prediction is not None:
#         result_y = GRID_OFFSET_Y + 250
        
#         # Draw "Prediction:" label
#         label_surface = button_font.render("Prediction:", True, TEXT_PRIMARY)
#         screen.blit(label_surface, (
#             classify_button.rect.x + BUTTON_WIDTH // 2 - label_surface.get_width() // 2,
#             result_y
#         ))
        
#         # Draw the predicted digit
#         digit_surface = result_font.render(str(prediction), True, GREEN)
#         screen.blit(digit_surface, (
#             classify_button.rect.x + BUTTON_WIDTH // 2 - digit_surface.get_width() // 2,
#             result_y + 40
#         ))
        
#         # Draw confidence
#         confidence_text = f"Confidence: {confidence*100:.1f}%"
#         confidence_surface = info_font.render(confidence_text, True, TEXT_SECONDARY)
#         screen.blit(confidence_surface, (
#             classify_button.rect.x + BUTTON_WIDTH // 2 - confidence_surface.get_width() // 2,
#             result_y + 110
#         ))
    
#     # Update display
#     pygame.display.flip()
#     clock.tick(60)

# pygame.quit()
# print("\nGoodbye!")







import sys
import os
import pygame
import numpy as np
from tkinter import Tk, filedialog

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Error: PyTorch is not installed.")
    print("Please install it with: pip install torch")
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
CELL_SIZE = 16  # 32x32 grid with 16px cells = 512x512 drawing area
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 50

# Colors (Dark style)
BACKGROUND = (30, 30, 30)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRID_BG = (45, 45, 45)
GRID_LINE = (70, 70, 70)
TEXT_PRIMARY = (255, 255, 255)  # Pure white
TEXT_SECONDARY = (255, 255, 255)  # Pure white
BLUE = (66, 135, 245)
RED = (220, 53, 69)
GREEN = (40, 200, 80)
PURPLE = (156, 39, 176)

# Button dimensions
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 50
BUTTON_MARGIN = 20

# Define the CNN model (same as in training)
class HandwritingCNN(nn.Module):
    def __init__(self):
        super(HandwritingCNN, self).__init__()
        
        # First convolutional layer - 32 filters, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer - 64 filters, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
        # Activation functions
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Load the trained model
print("Loading trained model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandwritingCNN().to(device)

try:
    checkpoint = torch.load("handwriting_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please train the model first by running: python train_pytorch.py")
    sys.exit(1)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Handwriting Recognition - PyTorch - Dark Mode + Image Loading")

# Fonts
title_font = pygame.font.Font(None, 48)
button_font = pygame.font.Font(None, 32)
result_font = pygame.font.Font(None, 64)
info_font = pygame.font.Font(None, 24)
small_font = pygame.font.Font(None, 20)

# Create the drawing grid (32x32)
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

# Prediction result
prediction = None
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
            
            # Draw cell with intensity based on grid value (white drawing on dark background)
            intensity = int(grid[row, col] * 255)
            # Clamp values to ensure they're within 0-255 range
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
    
    # Check if click is within grid
    if (GRID_OFFSET_X <= mouse_x < GRID_OFFSET_X + GRID_SIZE * CELL_SIZE and
        GRID_OFFSET_Y <= mouse_y < GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE):
        
        # Calculate grid position
        grid_x = (mouse_x - GRID_OFFSET_X) // CELL_SIZE
        grid_y = (mouse_y - GRID_OFFSET_Y) // CELL_SIZE
        
        # Paint with brush (paint surrounding cells too for smoother drawing)
        for dy in range(-brush_size + 1, brush_size):
            for dx in range(-brush_size + 1, brush_size):
                new_x = grid_x + dx
                new_y = grid_y + dy
                
                if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                    # Softer brush effect
                    distance = abs(dx) + abs(dy)
                    intensity = max(0, 1.0 - distance * 0.3)
                    grid[new_y, new_x] = min(1.0, grid[new_y, new_x] + intensity)

def load_image_to_grid():
    """Load an image file and convert it to the 32x32 grid"""
    global grid, loaded_image_path, prediction, confidence
    
    # Hide the pygame window temporarily
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # Open file dialog
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
            # Load the image
            img = Image.open(file_path)
            
            # Convert to grayscale
            img_gray = img.convert('L')
            
            # Resize to 32x32
            img_resized = img_gray.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            
            # Invert if needed (assuming white background with black digit)
            # Check if the image has more white than black (average > 0.5)
            if np.mean(img_array) > 0.5:
                img_array = 1.0 - img_array
            
            # Update grid
            grid[:] = img_array
            loaded_image_path = os.path.basename(file_path)
            
            # Reset prediction
            prediction = None
            confidence = None
            
            print(f"Loaded image: {file_path}")
            
        except Exception as e:
            print(f"Error loading image: {e}")
            loaded_image_path = None

def classify_digit():
    """Classify the drawn digit using PyTorch model"""
    global prediction, confidence
    
    # Check if grid has any content
    if np.max(grid) < 0.1:
        prediction = None
        confidence = None
        return
    
    # Resize 32x32 to 28x28 for the model
    from scipy import ndimage
    
    # Find bounding box and center it
    rows = np.any(grid > 0.1, axis=1)
    cols = np.any(grid > 0.1, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        prediction = None
        confidence = None
        return
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Extract the digit
    digit = grid[rmin:rmax+1, cmin:cmax+1]
    
    # Resize to fit in 20x20 box (leaving 4px border like MNIST)
    height, width = digit.shape
    scale = min(20.0 / height, 20.0 / width)
    
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # Resize the digit
    digit_resized = ndimage.zoom(digit, (new_height / height, new_width / width))
    
    # Create 28x28 image with centered digit
    image_28x28 = np.zeros((28, 28), dtype=np.float32)
    
    # Center it
    y_offset = (28 - new_height) // 2
    x_offset = (28 - new_width) // 2
    
    image_28x28[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = digit_resized
    
    # Reshape for PyTorch model input (batch_size, channels, height, width)
    model_input = torch.from_numpy(image_28x28).unsqueeze(0).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(model_input)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction_tensor = torch.argmax(probabilities, dim=1)
        confidence_tensor = probabilities[0][prediction_tensor]
        
        prediction = prediction_tensor.item()
        confidence = confidence_tensor.item()

def reset_grid():
    """Reset the drawing grid"""
    global grid, prediction, confidence, loaded_image_path
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    prediction = None
    confidence = None
    loaded_image_path = None

# Main game loop
running = True
clock = pygame.time.Clock()
drawing = False

print("\nInstructions:")
print("- Draw a digit (0-9) on the grid by clicking and dragging")
print("- OR click 'Load Image' to load an image file")
print("- Click 'Classify' to recognize the digit")
print("- Click 'Reset' to clear the grid")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                pos = pygame.mouse.get_pos()
                
                # Check button clicks
                if classify_button.is_clicked(pos):
                    classify_digit()
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
    
    # Clear screen with dark background
    screen.fill(BACKGROUND)
    
    # Draw title
    title_surface = title_font.render("Handwriting Recognition (PyTorch)", True, TEXT_PRIMARY)
    screen.blit(title_surface, (WINDOW_WIDTH // 2 - title_surface.get_width() // 2, 10))
    
    # Draw instructions
    instruction_surface = info_font.render("Draw a digit or load an image", True, TEXT_SECONDARY)
    screen.blit(instruction_surface, (GRID_OFFSET_X, GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE + 20))
    
    # Show loaded image name if applicable
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
    if prediction is not None:
        result_y = GRID_OFFSET_Y + 250
        
        # Draw "Prediction:" label
        label_surface = button_font.render("Prediction:", True, TEXT_PRIMARY)
        screen.blit(label_surface, (
            classify_button.rect.x + BUTTON_WIDTH // 2 - label_surface.get_width() // 2,
            result_y
        ))
        
        # Draw the predicted digit
        digit_surface = result_font.render(str(prediction), True, GREEN)
        screen.blit(digit_surface, (
            classify_button.rect.x + BUTTON_WIDTH // 2 - digit_surface.get_width() // 2,
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
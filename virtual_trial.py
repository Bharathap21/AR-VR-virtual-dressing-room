import os
import cv2
import math
import numpy as np
import mediapipe as mp 
from imutils import rotate_bound

# Initialize global states
ACTIVE_IMAGES = [0 for _ in range(10)]
SPRITES = [0 for _ in range(10)]
IMAGES = {i: [] for i in range(10)}
PHOTOS = {i: [] for i in range(10)}

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# ---------- Utility Functions ----------

def initialize_images_and_photos(file_path):
    """Initialize images with better error handling"""
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False
            
        idx_str = os.path.basename(file_path)
        digits = ''.join(filter(str.isdigit, idx_str))
        
        if not digits:
            print(f"No digits found in filename: {idx_str}")
            return False
            
        idx = int(digits)
        idx = (idx // 10) % 10
        
        print(f"Processing file: {file_path}")
        print(f"Extracted digits: {digits}")
        print(f"Category index: {idx}")
        
        sprite_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if sprite_image is None:
            print(f"Failed to load image: {file_path}")
            return False
            
        # Ensure the image has 4 channels (RGBA) for transparency
        if sprite_image.shape[2] == 3:
            # Add alpha channel if missing
            sprite_image = cv2.cvtColor(sprite_image, cv2.COLOR_BGR2BGRA)
            sprite_image[:, :, 3] = 255  # Full opacity
        
        # Don't clear existing images - append instead for multiple sprites per category
        if idx not in IMAGES:
            IMAGES[idx] = []
            PHOTOS[idx] = []
        
        IMAGES[idx].append(sprite_image)
        photo = cv2.resize(sprite_image, (150, 100))
        PHOTOS[idx].append(photo)
        
        print(f"Successfully loaded sprite for category {idx}: {file_path}")
        print(f"Total sprites in category {idx}: {len(IMAGES[idx])}")
        return True
        
    except Exception as e:
        print(f"Error initializing images: {e}")
        return False

def put_sprite(num, k):
    """Set active sprite with better validation"""
    try:
        if num < 0 or num >= 10:
            print(f"Invalid sprite category: {num}")
            return False
            
        print(f"Activating sprite category {num}")
        SPRITES[num] = 1
        
        # Ensure k is within bounds of available images
        if num in IMAGES and len(IMAGES[num]) > 0:
            ACTIVE_IMAGES[num] = k % len(IMAGES[num])
            print(f"Set active sprite: category {num}, index {ACTIVE_IMAGES[num]}")
            print(f"Available images in category {num}: {len(IMAGES[num])}")
            return True
        else:
            ACTIVE_IMAGES[num] = 0
            print(f"No images available for category {num}")
            return False
            
    except Exception as e:
        print(f"Error in put_sprite: {e}")
        return False

def calculate_inclination(p1, p2):
    dx = p2[0] - p1[0]
    if dx == 0:
        return 0
    return math.degrees(math.atan((p2[1] - p1[1]) / dx))

def adjust_sprite_to_head(sprite, head_width, head_ypos, scale_factor=1.0, ontop=True):
    if sprite is None or sprite.shape[1] == 0:
        return None, 0
    
    # Calculate scaling factor based on head width with additional scale control
    target_width = int(head_width * scale_factor)
    factor = target_width / sprite.shape[1]
    
    # Limit the scaling factor to prevent extremely large sprites
    factor = min(factor, 3.0)  # Increased max scaling for tiaras
    factor = max(factor, 0.1)  # Min 0.1x scaling
    
    sprite = cv2.resize(sprite, (0, 0), fx=factor, fy=factor)
    y_orig = head_ypos - sprite.shape[0] if ontop else head_ypos
    if y_orig < 0:
        sprite = sprite[abs(y_orig):, :, :]
        y_orig = 0
    return sprite, y_orig

def draw_sprite(frame, sprite, x_offset, y_offset):
    """Draw sprite with comprehensive error handling"""
    try:
        if sprite is None or sprite.size == 0:
            return frame
            
        if frame is None or frame.size == 0:
            print("Invalid frame for sprite drawing")
            return frame
            
        h, w = sprite.shape[:2]
        imgH, imgW = frame.shape[:2]

        # Validate coordinates
        if x_offset >= imgW or y_offset >= imgH or x_offset + w <= 0 or y_offset + h <= 0:
            print(f"Sprite out of bounds: x={x_offset}, y={y_offset}, w={w}, h={h}, frame={imgW}x{imgH}")
            return frame

        # Calculate safe boundaries
        start_y = max(0, y_offset)
        start_x = max(0, x_offset)
        end_y = min(imgH, y_offset + h)
        end_x = min(imgW, x_offset + w)
        
        # Calculate sprite crop boundaries
        sprite_start_y = max(0, -y_offset)
        sprite_start_x = max(0, -x_offset)
        sprite_end_y = sprite_start_y + (end_y - start_y)
        sprite_end_x = sprite_start_x + (end_x - start_x)
        
        # Crop sprite to fit
        cropped_sprite = sprite[sprite_start_y:sprite_end_y, sprite_start_x:sprite_end_x]
        
        if cropped_sprite.size == 0:
            return frame

        # Check if sprite has alpha channel
        if cropped_sprite.shape[2] == 4:
            # Alpha blending
            alpha = cropped_sprite[:, :, 3:4] / 255.0
            overlay = cropped_sprite[:, :, :3]
            background = frame[start_y:end_y, start_x:end_x]
            
            # Ensure shapes match
            if overlay.shape[:2] == background.shape[:2]:
                frame[start_y:end_y, start_x:end_x] = (
                    alpha * overlay + (1 - alpha) * background
                ).astype(np.uint8)
        else:
            # No alpha channel, direct copy
            if cropped_sprite.shape[:2] == frame[start_y:end_y, start_x:end_x].shape[:2]:
                frame[start_y:end_y, start_x:end_x] = cropped_sprite
        
        return frame
        
    except Exception as e:
        print(f"Error in draw_sprite: {e}")
        return frame

def apply_sprite(frame, sprite, head_width, x, y, angle, scale_factor=1.0, ontop=True):
    if sprite is None:
        return
        
    try:
        sprite = rotate_bound(sprite, angle)
        sprite, y_final = adjust_sprite_to_head(sprite, head_width, y, scale_factor, ontop)
        if sprite is not None:
            print(f"Applying sprite at position: x={x}, y={y_final}, scale={scale_factor}")
            draw_sprite(frame, sprite, x, y_final)
    except Exception as e:
        print(f"Error applying sprite: {e}")

def get_category_number(file_path):
    filename = os.path.basename(file_path)
    if filename:
        digits = ''.join(filter(str.isdigit, filename))
        if digits:
            val = int(digits)
            category = (val // 10) % 10
            print(f"File: {filename}, Digits: {digits}, Category: {category}")
            return category
    return -1

def get_k(file_path):
    filename = os.path.basename(file_path)
    if filename:
        digits = ''.join(filter(str.isdigit, filename))
        if digits:
            val = int(digits)
            k = val % 10
            print(f"File: {filename}, Digits: {digits}, K: {k}")
            return k
    return -1

def safe_apply_sprite(frame, sprite_category, head_width, x, y, angle, scale_factor=1.0, ontop=True):
    """Safely apply sprite with bounds checking"""
    print(f"Attempting to apply sprite category {sprite_category}")
    print(f"SPRITES[{sprite_category}] = {SPRITES[sprite_category]}")
    print(f"Images available: {sprite_category in IMAGES and len(IMAGES[sprite_category]) > 0}")
    
    if (sprite_category in IMAGES and 
        len(IMAGES[sprite_category]) > 0 and 
        ACTIVE_IMAGES[sprite_category] < len(IMAGES[sprite_category])):
        sprite = IMAGES[sprite_category][ACTIVE_IMAGES[sprite_category]]
        print(f"Applying sprite from category {sprite_category}, index {ACTIVE_IMAGES[sprite_category]}")
        apply_sprite(frame, sprite, head_width, x, y, angle, scale_factor, ontop)
        return True
    else:
        print(f"Cannot apply sprite category {sprite_category}: no images available or sprite not active")
        return False

# ---------- Main Frame Processing ----------

def process_frame(frame, file_path):
    global SPRITES, ACTIVE_IMAGES, IMAGES
    sprite_applied = False

    try:
        if isinstance(frame, bytes):
            frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)

        if frame is None or not hasattr(frame, 'shape'):
            print("Invalid frame")
            return None

        # Initialize the sprite from file
        if not initialize_images_and_photos(file_path):
            print("Failed to initialize images")
            return None

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            print("No face landmarks detected")
            return None

        for landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark])

            # Get key facial landmarks
            left = points[234]   # Left face boundary
            right = points[454]  # Right face boundary
            face_width = abs(right[0] - left[0])
            face_height = abs(points[10][1] - points[152][1])  # forehead to chin
            
            # Key facial landmarks with better positioning
            forehead = points[10]   # Top of forehead
            left_eyebrow = points[70]
            right_eyebrow = points[107]
            nose_tip = points[1]
            chin = points[152]
            left_ear = points[234]
            right_ear = points[454]
            neck_base = points[18]  # Base of neck
            
            # Calculate head center for better positioning
            head_center_x = (left[0] + right[0]) // 2
            head_center_y = (forehead[1] + chin[1]) // 2

            incl = calculate_inclination(left, right)
            is_mouth_open = (points[13][1] - points[14][1]) >= 10

            index = get_category_number(file_path)
            k = get_k(file_path)
            
            print(f"Processing sprite: category={index}, k={k}")
            
            if index >= 0:
                put_sprite(index, k)

            # Apply accessories with improved positioning and scaling
            
            # Tiara - positioned above forehead with better positioning
            if SPRITES[3]:
                tiara_x = head_center_x - int(face_width * 1)
                tiara_y = forehead[1] + int(face_height * 0.2)  # Reduced offset for better visibility
                print(f"Applying tiara at x={tiara_x}, y={tiara_y}")
                if safe_apply_sprite(frame, 3, face_width * 1.0, 
                                   tiara_x, tiara_y, 
                                   incl, scale_factor=2, ontop=True):
                    sprite_applied = True
                    print("Tiara applied successfully")

            # Necklace - positioned at neck base
            if SPRITES[1] and safe_apply_sprite(frame, 1, face_width * 0.6, 
                                              head_center_x - int(face_width * 0.55), 
                                              neck_base[1] + int(face_height * 0.1), 
                                              incl, scale_factor=2, ontop=False):
                sprite_applied = True

            # Goggles - positioned over eyes
            if SPRITES[6] and safe_apply_sprite(frame, 6, face_width * 0.7, 
                                              head_center_x - int(face_width * 0.5), 
                                              head_center_y - int(face_height * 0.5), 
                                              incl, scale_factor=1.5, ontop=False):
                sprite_applied = True

            # Earrings - positioned at ears
            if SPRITES[2]:
                earring_scale = 0.2
                if safe_apply_sprite(frame, 2, face_width * 0.15, 
                                   left_ear[0] - int(face_width * 0.1), 
                                   left_ear[1], 
                                   incl, scale_factor=earring_scale, ontop=False):
                    sprite_applied = True
                if safe_apply_sprite(frame, 2, face_width * 0.15, 
                                   right_ear[0] + int(face_width * 0.05), 
                                   right_ear[1], 
                                   incl, scale_factor=earring_scale, ontop=False):
                    sprite_applied = True

            # Tops/Clothing - positioned at shoulders/chest
            if SPRITES[4] and safe_apply_sprite(frame, 4, face_width * 1.5, 
                                              head_center_x - int(face_width * 2.5), 
                                              neck_base[1] + int(face_height * 0), 
                                              incl, scale_factor=3, ontop=False):
                sprite_applied = True

        print(f"Frame processing complete. Sprite applied: {sprite_applied}")
        return frame if sprite_applied else None
        
    except Exception as e:
        print(f"Error in process_frame: {e}")
        return None
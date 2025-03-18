import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from boundaries import *

DEBUG = False

class SpriteDetector:
    def __init__(self, sprites_dir: str):
        """
        Initialize detector with sprites directory.
        Expects subdirectories for each element type (enemies, blocks, etc.)
        """
        self.templates: Dict[str, List[Tuple[np.ndarray, str]]] = {}
        self.load_sprites(sprites_dir)
        
        # Use different thresholds for different sprite categories
        self.thresholds = {
            "enemies": 0.73,  # Lower threshold for enemies to account for variations
            "blocks": 0.85,
            "structures": 0.95
        }
        # Default threshold for any category not explicitly specified
        self.default_threshold = 0.8

    def load_sprites(self, sprites_dir: str):
        """
        Load sprite templates from subdirectories.
        Each subdirectory name becomes a category.
        """
        for category_dir in Path(sprites_dir).iterdir():
            #print(category_dir)
            if category_dir.is_dir():
                category = category_dir.name
                self.templates[category] = []
                
                # Load all PNG files in the category directory
                for sprite_file in category_dir.glob('*.png'):
                    #print("\t",sprite_file)
                    template = cv2.imread(str(sprite_file))
                    if template is not None:
                        # Store both color and grayscale versions for flexibility
                        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                        self.templates[category].append((template, gray_template, sprite_file.stem))

    def detect_sprite(self, image: np.ndarray, template: np.ndarray, category: str, sprite_name: str) -> bool:
        """
        Detect if a specific template appears in the image using template matching.
        Returns True if template is found with confidence above threshold.
        """
        threshold = self.thresholds.get(category, self.default_threshold)
        
        # For enemies, use a more forgiving approach
        if category == "enemies":
            # Try both regular template matching and another method that ignores background
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Regular template matching
            result1 = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            
            # Method 2: Edge-based matching to reduce background influence
            # Detect edges in both the image and template
            image_edges = cv2.Canny(gray_image, 50, 150)
            template_edges = cv2.Canny(template, 50, 150)
            
            # Match on edges
            w, h = template_edges.shape[::-1]
            if w < image_edges.shape[1] and h < image_edges.shape[0]:  # Ensure template fits in image
                result2 = cv2.matchTemplate(image_edges, template_edges, cv2.TM_CCOEFF_NORMED)
                
                # Return true if either method finds a match
                return np.max(result1) >= threshold or np.max(result2) >= threshold * 0.9
            return np.max(result1) >= threshold
        else:
            # For non-enemy sprites, use standard template matching
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            return np.max(result) >= threshold

    def detect_sprites_in_category(self, image: np.ndarray, category: str) -> List[str]:
        """
        Detect all sprites from a category in the image.
        Returns list of detected sprite names.
        """
        detected = set()
        for color_template, gray_template, sprite_name in self.templates[category]:
            if self.detect_sprite(image, gray_template, category, sprite_name):
                if DEBUG: print(f"Simple match: {sprite_name}")
                base_name = sprite_name.split('_')[0]  # Extract base name before underscore
                if base_name != "floor": # Floor handled differently by analyze_floor
                    detected.add(base_name)
        
        return list(detected)

class EnhancedSpriteDetector(SpriteDetector):
    def __init__(self, sprites_dir: str):
        """
        Initialize enhanced detector with sprites directory.
        Extends the base SpriteDetector with better description capabilities.
        """
        super().__init__(sprites_dir)
        
        # Extra distance threshold to detect patterns
        self.pattern_distance_threshold = 32  # Pixels
        
    def detect_pattern(self, sprite_locations, image_width, sprite_type = None):
        """
        Analyze the pattern of sprite locations.
        Returns a string describing the pattern.
        
        Args:
            sprite_locations: List of (x, y) coordinates where sprites were detected
            image_width: Width of the image for horizontal pattern detection
        """
        if len(sprite_locations) <= 1:
            return ""  # No pattern with just one instance
            
        # Sort locations by x-coordinate
        sorted_locs = sorted(sprite_locations, key=lambda loc: loc[0])
        
        # Horizontal girders are actually just parts of the same girder. More brick sprites in a row needed to make "three"
        if len(sorted_locs) >= 3 and sprite_type != "girder" and (sprite_type != "brickblock" or len(sorted_locs) >= 4):
            # Check for horizontal line (same y-coordinate, evenly spaced x)
            y_values = [loc[1] for loc in sorted_locs]
            if max(y_values) - min(y_values) < 5:  # All at similar height
                # Check if evenly spaced
                x_diffs = [sorted_locs[i+1][0] - sorted_locs[i][0] for i in range(len(sorted_locs)-1)]
                if max(x_diffs) - min(x_diffs) < 2 and min(x_diffs) > 0 and max(x_diffs) < 20:
                    return "in a horizontal line"
                
        # Check for vertical line (same x-coordinate, closely spaced y)
        x_values = [loc[0] for loc in sorted_locs]
        if len(sorted_locs) >= 3 and max(x_values) - min(x_values) < 5:  # All at similar x-position
            # Check if closely and evenly spaced
            sorted_by_y = sorted(sprite_locations, key=lambda loc: loc[1])
            y_diffs = [sorted_by_y[i+1][1] - sorted_by_y[i][1] for i in range(len(sorted_by_y)-1)]
            if max(y_diffs) - min(y_diffs) < 2 and min(y_diffs) > 0 and max(y_diffs) < 20:
                return "in a vertical line"
                            
        # Check for cluster (close together but not in a clear pattern)
        all_distances = []
        for i in range(len(sprite_locations)):
            for j in range(i+1, len(sprite_locations)):
                x1, y1 = sprite_locations[i]
                x2, y2 = sprite_locations[j]
                distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
                all_distances.append(distance)
            
        # Don't let brickledges or girders cluster    
        #if sprite_type != "brickledge" and sprite_type != "girder":
        if sprite_type != "girder":
            avg_distance = sum(all_distances) / len(all_distances) if all_distances else float('inf')
            if avg_distance < self.pattern_distance_threshold:
                return "clustered"
            
        # Default if no clear pattern detected
        return ""
    
    def detect_sprites_in_category_enhanced(self, image: np.ndarray, category: str) -> List[str]:
        """
        Enhanced detection with quantity and pattern recognition.
        Returns more descriptive strings about the detected sprites.
        """
        # First, get locations of all sprite matches
        sprite_locations = {}
        height, width = image.shape[:2]
        
        for color_template, gray_template, sprite_name in self.templates[category]:
            # Extract base name (e.g., "coin" from "coin_spinning")
            if '_' in sprite_name:
                base_name = sprite_name.split('_')[0]
            else:
                base_name = sprite_name
                
            if DEBUG: print(f"\tDetecting {sprite_name}")

            # Skip if not relevant for quantity analysis
            if base_name not in ["coin", "brickblock", "mushroom", "wood", "solidblock", "questionblock", "goomba", "koopa", "bill", "helmet", "hammerturtle", "plant", "spiny", # "brickledge", 
                                 "cannon", "metal", "greenpipe", "whitepipe", # "obstacle", 
                                 "girder", "tree", "bush", "cloud", "hill", "staircase"]:
                continue
                
            # Find all instances using template matching
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold = self.thresholds.get(category, self.default_threshold)
            
            # Use template matching method suited for the category
            method = cv2.TM_CCOEFF_NORMED
            
            result = cv2.matchTemplate(gray_image, gray_template, method)
            locations = np.where(result >= threshold)
            
            # Process matched locations
            template_h, template_w = gray_template.shape
            
            # Initialize sprite type in dictionary if not exists
            if base_name not in sprite_locations:
                sprite_locations[base_name] = []
                
            # Add all matched locations, with non-max suppression to avoid duplicates
            for pt in zip(*locations[::-1]):  # Convert from (y,x) to (x,y)
                # Calculate center coordinates
                center_x = pt[0] + template_w // 2
                center_y = pt[1] + template_h // 2
    
                # Check if this point is too close to an already detected point
                too_close = False
                for existing_x, existing_y in sprite_locations[base_name]:
                    if abs(center_x - existing_x) < template_w/2 and abs(center_y - existing_y) < template_h/2:
                        too_close = True
                        break
    
                if not too_close:
                    sprite_locations[base_name].append((center_x, center_y))
                    if DEBUG: print(f"\t\t{base_name} at center ({center_x}, {center_y})")
     
        # Process results into descriptions
        descriptions = []
        
        for sprite_type, locations in sprite_locations.items():
            # Skip if none detected
            if not locations:
                continue
                
            count = len(locations)
            if sprite_type in ["girder", "mushroom", "wood"]:
                count = len(set([y for x, y in locations]))  # Count unique y-values

            if sprite_type == "staircase" and count > 0:
                    threshold = 48
                    # Sort by x-coordinate
                    locations.sort(key=lambda loc: loc[0])

                    grouped_locations = []
                    current_group = [locations[0]]

                    for x, y in locations[1:]:
                        prev_x, _ = current_group[-1]
                        if x - prev_x < threshold:
                            current_group.append((x, y))
                        else:
                            # Compute average for the completed group
                            avg_x = sum(loc[0] for loc in current_group) / len(current_group)
                            avg_y = sum(loc[1] for loc in current_group) / len(current_group)
                            grouped_locations.append((avg_x, avg_y))
                            current_group = [(x, y)]

                    # Don't forget the last group
                    avg_x = sum(loc[0] for loc in current_group) / len(current_group)
                    avg_y = sum(loc[1] for loc in current_group) / len(current_group)
                    grouped_locations.append((avg_x, avg_y))

                    count = len(grouped_locations)
                    locations = grouped_locations
            
            display_name = get_sprite_display_name(sprite_type)

            if count > 0:
                # Get quantity description
                if count == 1:
                    quantity = f"a {display_name}"
                elif count == 2:
                    quantity = f"two {display_name}s"
                elif count == 3:
                    quantity = f"three {display_name}s"
                else:
                    quantity = f"several {display_name}s"
                
                # Get pattern description if more than one
                pattern = ""
                if count > 1:
                    pattern = self.detect_pattern(locations, width, sprite_type)
                    if pattern:
                        pattern = " " + pattern
                        
                    loc_phrases = [location_description_in_image(image, v, sprite_type) for v in locations]
                    loc_phrases = set(loc_phrases) # Eliminates duplicates
                    if len(loc_phrases) == 1: # They are all in the same area
                        pattern = pattern + " in the " + list(loc_phrases)[0] # Convert back to list to access an index
                    elif "clustered" in pattern: # or "line" in pattern: # Refer to clustered elements with a single location, even if they straddle a border
                        pattern = pattern + " in the " + location_description_in_image(image, average_point(locations), sprite_type)
                    else: # len(loc_phrases) == count: # Each is in a different area
                        all_locations = location_phrase_sort(list(loc_phrases))
                        pattern = pattern + " in the " + (" and ".join(all_locations))
                elif count == 1:
                    pattern = " in the " + location_description_in_image(image, locations[0], sprite_type)
                
                descriptions.append(f"{quantity}{pattern}")
        
        # For enemies category, include enemy count
        if category == "enemies":
            # Special handling for enemy descriptions
            return self._format_enemy_count_enhanced(image, sprite_locations)
            
        # Add any additional sprites from the original method that we might have missed
        simple_detected = super().detect_sprites_in_category(image, category)
        
        # Filter out anything we've already described
        for detected in simple_detected:
            # Check if this is a base type we've already covered
            already_covered = False
            for sprite_type in sprite_locations.keys():
                if detected == sprite_type or detected.startswith(sprite_type):
                    already_covered = True
                    break
            
            if not already_covered:
                descriptions.append(get_sprite_display_name(detected))
        
        return descriptions
    
    def _format_enemy_count_enhanced(self, image, enemy_locations: Dict[str, List[Tuple[int, int]]]) -> List[str]:
        """Format enhanced enemy descriptions with counts and patterns."""
        result = []
        
        for enemy_type, locations in enemy_locations.items():
            count = len(locations)
            
            display_name = get_sprite_display_name(enemy_type)

            if count > 0:
                if count == 1:
                    result.append(f"a {display_name}")
                elif count == 2:
                    result.append(f"two {display_name}s")
                elif count == 3:
                    result.append(f"three {display_name}s")
                else:
                    result.append(f"several {display_name}s")
            
            # Add pattern for multiple enemies
            if count > 1:
                pattern = self.detect_pattern(locations, 256)  # Assuming typical screen width
                if pattern:
                    result[-1] += f" {pattern}"

                loc_phrases = [location_description_in_image(image, v, enemy_type) for v in locations]
                loc_phrases = set(loc_phrases) # Eliminates duplicates
                if len(loc_phrases) == 1: # They are all in the same area
                    result[-1] += " in the " + list(loc_phrases)[0] # Convert back to list to access an index
                else: # if len(loc_phrases) == count: # Each is in a different area
                    result[-1] += " in the " + (" and ".join(location_phrase_sort(list(loc_phrases))))

            elif count == 1:
                result[-1] += " in the " + location_description_in_image(image, locations[0], enemy_type)

        return result

def location_phrase_sort(loc_phrases):
    order = [
        "top left",
        "top center",
        "top right",
        "center left",
        "center",
        "center right",
        "bottom left",
        "bottom center",
        "bottom right"
    ]
    
    # Create a dictionary mapping each phrase to its index in the order list
    order_dict = {phrase: index for index, phrase in enumerate(order)}
    
    # Sort loc_phrases based on the order dictionary
    return sorted(loc_phrases, key=lambda x: order_dict.get(x, float('inf')))

def get_sprite_display_name(sprite_type):
    display_name = sprite_type
    # Name changes and adding of spaces in names for certain sprites
    if sprite_type == "questionblock":
        display_name = "question block"
    elif sprite_type == "greenpipe":
        display_name = "green pipe"
    elif sprite_type == "whitepipe":
        display_name = "white pipe"
    elif sprite_type == "whitepipe":
        display_name = "white pipe"
    #elif sprite_type == "brickledge":
    #    display_name = "brick ledge"
    elif sprite_type == "brickblock":
        display_name = "brick block"
    elif sprite_type == "solidblock":
        display_name = "solid block"
    elif sprite_type == "metal":
        display_name = "metal block"
    #elif sprite_type == "obstacle":
    #    display_name = "vertical obstacle"
    elif sprite_type == "bill":
        display_name = "bullet bill"
    elif sprite_type == "plant":
        display_name = "piranha plant"
    elif sprite_type == "hammerturtle":
        display_name = "hammer bro"
    elif sprite_type == "ceiling":
        display_name = "brick ceiling"
    elif sprite_type == "mushroom":
        display_name = "giant mushroom platform"
    elif sprite_type == "wood":
        display_name = "giant tree platform"
    
    return display_name

def get_floor_template(detector: SpriteDetector) -> np.ndarray:
    """
    Find the floor template from the detector's loaded templates.
    """
    for category, templates in detector.templates.items():
        if category == "blocks":
            for color_template, gray_template, sprite_name in templates:
                # Look for floor_1 or any sprite name starting with "floor_"
                if sprite_name == "floor_1" or sprite_name.startswith("floor_"):
                    return color_template
    
    # Return None if floor template not found
    return None

def analyze_floor(image: np.ndarray, floor_template: np.ndarray, background_color=None, is_shifted=False) -> str:
    """
    Analyze the floor in the image using the floor template and detect gaps.
    
    Args:
        image: The full screenshot image
        floor_template: The floor block template image
        background_color: The background color to identify gaps (if None, detect automatically)
        is_shifted: If True, then there is a floor, but it is not aligned with the bottom of the screen
    
    Returns:
        String description of the floor: "full floor", "floor with gaps", or "no floor"
    """
    # Extract bottom strip (16 pixels high)
    height, width = image.shape[:2]
    bottom_strip = image[height-16:height, :]
    
    # If we're dealing with a shifted floor, we just need to check for background color gaps
    if is_shifted:
        if DEBUG: print("shifted floor")
        # Automatically detect background color if not provided
        if background_color is None:
            # In Mario, sky/background is usually the top-left pixel
            background_color = image[0, 0].tolist()
        
        # Create binary mask where pixels match the background color
        tolerance = 30  # Allow some variation in color matching
        bg_mask = np.zeros((bottom_strip.shape[0], bottom_strip.shape[1]), dtype=np.uint8)
        
        for y in range(bottom_strip.shape[0]):
            for x in range(bottom_strip.shape[1]):
                pixel = bottom_strip[y, x].tolist()
                # Check if pixel color is close to background color
                if all(abs(pixel[i] - background_color[i]) < tolerance for i in range(3)):
                    bg_mask[y, x] = 255
        
        # Check for contiguous 5x5 regions of background color
        kernel = np.ones((5, 5), np.uint8)
        gaps = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)
        
        result = "floor with gaps" if np.any(gaps > 0) else "full floor"
        if DEBUG: print(f"Floor: {result}")
        return result    

    # For non-shifted floors, proceed with the original detection
    # Convert to grayscale for template matching
    gray_strip = cv2.cvtColor(bottom_strip, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(floor_template, cv2.COLOR_BGR2GRAY)
    
    # Check if floor blocks exist in the bottom strip
    result = cv2.matchTemplate(gray_strip, gray_template, cv2.TM_CCOEFF_NORMED)
    floor_exists = np.max(result) >= 0.8  # Using threshold of 0.8
    
    if not floor_exists:
        if DEBUG: print("no floor!")
        return "no floor"
    
    # Automatically detect background color if not provided
    if background_color is None:
        # In Mario, sky/background is usually the top-left pixel
        background_color = image[0, 0].tolist()
    
    # Create binary mask where pixels match the background color
    tolerance = 30  # Allow some variation in color matching
    bg_mask = np.zeros((bottom_strip.shape[0], bottom_strip.shape[1]), dtype=np.uint8)
    
    for y in range(bottom_strip.shape[0]):
        for x in range(bottom_strip.shape[1]):
            pixel = bottom_strip[y, x].tolist()
            # Check if pixel color is close to background color
            if all(abs(pixel[i] - background_color[i]) < tolerance for i in range(3)):
                bg_mask[y, x] = 255
    
    # Check for contiguous 5x5 regions of background color
    kernel = np.ones((5, 5), np.uint8)
    gaps = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)
    
    result = "floor with gaps" if np.any(gaps > 0) else "full floor"
    if DEBUG: print(f"Floor: {result}")
    return result

def format_caption(basic_props: Dict, detected_elements: Dict[str, List[str]], 
                  use_detailed_format: bool = False) -> str:
    """
    Format caption according to chosen structure.
    """
    if use_detailed_format:
        # Collect all elements
        all_elements = []
        for category, items in detected_elements.items():
            all_elements.extend(items)
        
        return (f"pixel art Super Mario Bros level, {basic_props['level_type']} stage, "
                f"{basic_props['sky_type']} sky background, {basic_props['floor']}, "
                f"{', '.join(all_elements) if all_elements else 'empty scene'}, "
                f"8-bit NES graphics, side-scrolling view")
    else:
        # Simple format similar to your original captions
        elements = [
            f"{basic_props['level_type']} level",
            f"{basic_props['sky_type']} sky",
            basic_props['floor']
        ]
        
        # Add all detected elements
        for category, items in detected_elements.items():
            elements.extend(items)
            
        return ". ".join(elements)

def format_caption_enhanced(basic_props: Dict, detected_elements: Dict[str, List[str]]) -> str:
    """
    Enhanced caption formatting with better element descriptions.
    """
    # Simple format similar to your original captions
    elements = [
        f"{basic_props['level_type']} level",
        f"{basic_props['sky_type']} sky",
        basic_props['floor']
    ]
        
    # Add all detected elements
    for category, items in detected_elements.items():
        elements.extend(items)
            
    return ". ".join(elements)

def process_directory_enhanced(input_dir: str, sprites_dir: str, output_file: str):
    """
    Process all screenshots in directory with enhanced sprite detection.
    """
    detector = EnhancedSpriteDetector(sprites_dir)
    
    with open(output_file, 'w') as f:
        for image_file in sorted(Path(input_dir).glob('*.png')):  # Sort files for consistent processing
            if DEBUG: print(f"Process {image_file}")
            # Read image
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"Failed to read {image_file}")
                continue
            
            # Get basic properties including floor detection
            floor_template = get_floor_template(detector)
            
            # Check for shifted floors based on level name patterns
            is_shifted = any(pattern in image_file.name for pattern in ['mario-1-3', 'mario-3-3'])
            
            # Get basic properties with the updated floor detection
            basic_props = generate_basic_caption_enhanced(image, image_file.name, detector, floor_template, is_shifted)
            
            # Detect sprites from each category with enhanced descriptions
            detected_elements = {}
            skip_image = False # Should this image even be in the training set?
            skip_reason = None
            for category in detector.templates.keys():
                detected = detector.detect_sprites_in_category_enhanced(image, category)
                # Don't include any training data that features these sprites
                for target in ["bridge", "flag", "spring", "vine", "helmet", "spiny", "water", "wall"]:
                    if any(target in s for s in detected):
                        skip_image = True
                        skip_reason = target
     
                if skip_image:
                    break

                if detected:
                    detected_elements[category] = detected
            
            if skip_image:
                if DEBUG: 
                    print(f"Skipping image {image_file} because it contains {skip_reason}")
                continue # Go to next candidate image
            
            # Generate enhanced caption
            caption = format_caption_enhanced(basic_props, detected_elements)
            
            # Create JSONL entry
            entry = {
                "file_name": image_file.name,
                "text": caption
            }
            
            # Write to file
            json.dump(entry, f)
            f.write('\n')
            
            # Print progress every 100 images
            if image_file.name.endswith('00.png'):
                print(f"Processed: {image_file.name}")

def generate_basic_caption_enhanced(image: np.ndarray, filename: str, detector: EnhancedSpriteDetector, 
                                   floor_template=None, is_shifted=False) -> Dict:
    """
    Generate basic image properties with enhanced floor detection.
    """
    # Determine level type from filename
    is_underworld = filename.startswith(('mario-1-2', 'mario-4-2'))
    level_type = "underworld" if is_underworld else "overworld"
    
    # Determine sky type from top pixels
    top_row = image[0:16, :]  # Check top 16 pixels
    avg_brightness = np.mean(top_row)
    sky_type = "blue" if avg_brightness > 128 else "night"
    
    # Get background color based on level type
    background_color = None
    if level_type == "overworld":
        background_color = image[5, 5].tolist()  # Sky color from top
    else:
        # For underworld, black background
        background_color = [0, 0, 0]
    
    # Get floor template if not provided
    if floor_template is None:
        floor_template = get_floor_template(detector)
    
    # Needs special handling since the floor does not align with bottom of screen in usual way
    is_shifted = filename.startswith('mario-8-3')

    # Analyze floor using the template
    floor_description = analyze_floor(image, floor_template, background_color, is_shifted)
    
    return {
        "level_type": level_type,
        "sky_type": sky_type,
        "floor": floor_description
    }

def get_floor_template(detector: SpriteDetector) -> np.ndarray:
    """
    Find the floor template from the detector's loaded templates.
    """
    for category, templates in detector.templates.items():
        if category == "blocks":
            for color_template, gray_template, sprite_name in templates:
                # Look for floor_1 or any sprite name starting with "floor_"
                if sprite_name == "floor_1" or sprite_name.startswith("floor_"):
                    return color_template
    
    # Return None if floor template not found
    return None

def location_description_in_image(image, location, sprite_type):
    """
        Treat level as 3 by 3 grid and assign names to each section
    """
    height, width = image.shape[:2]
    x,y = location

    left = x < LEFT_LINE
    right = x > RIGHT_LINE
    horizontal_center = not left and not right

    # Horizontal only
    if sprite_type in ["greenpipe", "whitepipe", "tree", "hill", "staircase"]:
        if left:
            return "left side"
        elif right:
            return "right side"
        elif horizontal_center:
            return "the middle"
        else:
            raise ValueError("How can this location not be horizontally placed? "+str(location))

    top = y < TOP_LINE
    bottom = y > BOTTOM_LINE
    vertical_center = not top and not bottom

    if top and left:
        return "top left"
    elif top and right:
        return "top right"
    elif bottom and left:
        return "bottom left"
    elif bottom and right:
        return "bottom right"
    elif left and vertical_center:
        return "center left"
    elif right and vertical_center:
        return "center right"
    elif top and horizontal_center:
        return "top center"
    elif bottom and horizontal_center:
        return "bottom center"
    elif vertical_center and horizontal_center:
        return "center"
    else:
        print("left",left)
        print("right",right)
        print("horizontal_center",horizontal_center)
        print("top",top)
        print("bottom",bottom)
        print("vertical_center",vertical_center)
        raise ValueError("How can this location not be placed? "+str(location))

def average_point(points):
    """
    Calculates the average point of a list of two-tuples.

    Args:
      points: A list of two-tuples representing points in 2D space.

    Returns:
      A tuple representing the average point, or None if the input list is empty.
    """
    if not points:
        return None

    x_sum = sum(x for x, y in points)
    y_sum = sum(y for x, y in points)

    return (x_sum / len(points), y_sum / len(points))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate captions for Mario screenshots")
    parser.add_argument("input_dir", help="Directory containing PNG screenshots")
    parser.add_argument("sprites_dir", help="Directory containing sprite templates")
    parser.add_argument("output_file", help="Output JSONL file path")

    args = parser.parse_args()
    process_directory_enhanced(args.input_dir, args.sprites_dir, args.output_file)

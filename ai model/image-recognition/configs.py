"""
Configuration file for the image recognition project.
Add these food detection configurations to your existing configs.py file.
"""

# Food detection configurations
FOOD_DETECTION = {
    # Model selection
    "use_clip": True,
    "use_object_detection": True,
    
    # Thresholds
    "clip_confidence_threshold": 0.25,
    "object_detection_threshold": 0.5,
    
    # Output settings
    "visualize_results": True,
    "save_visualizations": True,
    "visualization_output_dir": "data/processed/detection_results"
}

# Common food categories to detect
FOOD_CATEGORIES = [
    # Pasta and grains
    "spaghetti", "pasta", "rice", "quinoa", "oats", "cereal",
    
    # Vegetables
    "tomatoes", "cucumber", "lettuce", "romaine lettuce", "carrots", 
    "onions", "garlic", "bell pepper", "avocado", "potatoes",
    
    # Fruits
    "blueberries", "raspberries", "strawberries", "bananas", "apples", 
    "oranges", "lemons", "berries",
    
    # Proteins
    "chicken", "chicken breast", "beef", "ground beef", "bacon", "sausage", 
    "eggs", "tuna", "salmon", "shrimp",
    
    # Dairy
    "milk", "yogurt", "parmesan cheese", "cheddar cheese", "cheese", "butter", "cream",
    
    # Canned goods
    "canned tomatoes", "canned beans", "canned tuna", "canned corn",
    
    # Baked goods
    "bread", "bagels", "tortillas", 
    
    # Condiments and oils
    "olive oil", "vegetable oil", "salt", "pepper", "ketchup", "mustard",
    
    # Generic categories
    "fruits", "vegetables", "meat", "seafood"
]

# Pre-trained model paths
MODEL_PATHS = {
    "clip": "models/saved_models/clip_food_model",
    "object_detection": "models/saved_models/food_detection_model"
}
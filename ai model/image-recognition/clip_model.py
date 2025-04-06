import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pandas as pd
import os
from pathlib import Path

THRESHOLD = 0.05  # Default threshold value
TOP_K = 7  # Default top_k value

class FoodRecognizer:
    def __init__(self, model_name="openai/clip-vit-base-patch32",
                 ingredients_csv="data/top_500_ingredients.csv"):
        """
        Initialize the CLIP-based food ingredient recognizer.
        
        Args:
            model_name (str): HuggingFace model ID for CLIP
            ingredients_csv (str): Path to CSV containing 'ingredient' column.
                                   If None, tries to find it automatically.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        if ingredients_csv is None:
            current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            
            possible_paths = [
                current_dir / "data" / "top_500_ingredients.csv",
                current_dir.parent / "data" / "top_500_ingredients.csv",
                current_dir / ".." / "data" / "top_500_ingredients.csv",
                Path("ai_model") / "image-recognition" / "data" / "top_500_ingredients.csv",
                Path("ai model") / "image-recognition" / "data" / "top_500_ingredients.csv"
            ]
            
            for path in possible_paths:
                if path.exists():
                    ingredients_csv = str(path)
                    print(f"Found ingredients CSV at: {ingredients_csv}")
                    break
        
        # Load ingredients
        if ingredients_csv and os.path.exists(ingredients_csv):
            try:
                df = pd.read_csv(ingredients_csv)
                self.ingredients = df["ingredient"].tolist()
                print(f"Loaded {len(self.ingredients)} ingredients from {ingredients_csv}")
            except Exception as e:
                print(f"Error loading ingredients CSV: {e}")
                self.use_default_ingredients()
        else:
            print(f"Ingredients CSV not found at {ingredients_csv}, using default ingredients")
            self.use_default_ingredients()
    
    def use_default_ingredients(self):
        """Load a default set of common ingredients."""
        self.ingredients = [
            "tomato", "lettuce", "cucumber", "onion", "carrot", "potato", 
            "rice", "pasta", "spaghetti", "chicken", "beef", "pork", "fish",
            "egg", "milk", "cheese", "yogurt", "bread", "flour", "sugar",
            "salt", "pepper", "olive oil", "butter", "garlic", "avocado",
            "lemon", "lime", "apple", "banana", "orange", "strawberry", 
            "blueberry", "chocolate", "coffee", "tea", "canned tomatoes",
            "beans", "chickpeas", "corn", "frozen vegetables", "herbs",
            "spices", "nuts", "seeds"
        ]
        print(f"Using {len(self.ingredients)} default ingredients")
        
    def recognize_from_file(self, image_path, threshold=None, top_k=None):
        """
        Recognize food ingredients from an image file.
        
        Args:
            image_path (str): Path to the image file
            threshold (float): Confidence threshold for ingredient detection
            top_k (int): Maximum number of ingredients to return
            
        Returns:
            list[tuple]: Detected ingredient names with probabilities
        """
        threshold = threshold if threshold is not None else THRESHOLD
        top_k = top_k if top_k is not None else TOP_K
        
        try:
            image = Image.open(image_path).convert("RGB")
            print(f"Successfully loaded image from {image_path}")
            return self.recognize(image, threshold, top_k)
        except Exception as e:
            print(f"Error loading image from {image_path}: {e}")
            raise
    
    def recognize(self, image, threshold=None, top_k=None):
        """
        Recognize food ingredients from a PIL Image.
        
        Args:
            image (PIL.Image): Input image
            threshold (float): Confidence threshold for ingredient detection
            top_k (int): Maximum number of ingredients to return
            
        Returns:
            list[tuple]: Detected ingredient names with probabilities
        """
        threshold = threshold if threshold is not None else THRESHOLD
        top_k = top_k if top_k is not None else TOP_K
        
        # Create prompt templates for better CLIP performance
        texts = [f"a photo of {ingredient} food ingredient" for ingredient in self.ingredients]
        
        # Process inputs
        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get probabilities
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # Filter and sort results
        detected_ingredients = []
        for ingredient, prob in zip(self.ingredients, probs):
            if prob >= threshold:
                detected_ingredients.append((ingredient, float(prob)))

        detected_ingredients = sorted(detected_ingredients, 
                                      key=lambda x: x[1], 
                                      reverse=True)[:top_k]
        
        return detected_ingredients
    
    def recognize_batch(self, images, threshold=None, top_k=None):
        """
        Process multiple images in batch.
        
        Args:
            images (list[PIL.Image]): List of input images
            threshold (float): Confidence threshold for ingredient detection
            top_k (int): Maximum number of ingredients to return
            
        Returns:
            list[list[tuple]]: Detected ingredient names with probabilities for each image
        """
        threshold = threshold if threshold is not None else THRESHOLD
        top_k = top_k if top_k is not None else TOP_K
        
        results = []
        for image in images:
            results.append(self.recognize(image, threshold, top_k))
        return results

    def get_ingredients_array(self, image_path, threshold=None, top_k=None):
        """
        Get ingredients as a simple array of strings.
        
        Args:
            image_path (str): Path to the image file
            threshold (float): Confidence threshold for ingredient detection
            top_k (int): Maximum number of ingredients to return
            
        Returns:
            list[str]: List of detected ingredient names
        """
        results = self.recognize_from_file(image_path, threshold, top_k)
        return [ingredient for ingredient, _ in results]


def set_threshold(value):
    """Set the global threshold value."""
    global THRESHOLD
    THRESHOLD = value
    
def set_top_k(value):
    """Set the global top_k value."""
    global TOP_K
    TOP_K = value
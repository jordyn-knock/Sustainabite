# clip_model.py

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pandas as pd 
import ast

THRESHOLD = 0.02  # Default threshold value
TOP_K = 20  # Default top_k value

class FoodRecognizer:
    def __init__(self, model_name="openai/clip-vit-base-patch32",
                 ingredients_csv="data/top_500_ingredients.csv"):
        """
        Initialize the CLIP-based food ingredient recognizer.
        
        Args:
            model_name (str): HuggingFace model ID for CLIP
            ingredients_csv (str): Path to CSV containing 'ingredient' column 
                                   (defaults to your top_500_ingredients.csv).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        if ingredients_csv:
            df = pd.read_csv(ingredients_csv)
            self.ingredients = df["ingredient"].tolist()
        else:
            # Fallback if CSV not found or you want a smaller list for debugging
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
        
    def recognize_from_file(self, image_path, threshold=None, top_k=None):
        """
        Recognize food ingredients from an image file.
        
        Args:
            image_path (str): Path to the image file
            threshold (float): Confidence threshold for ingredient detection
            top_k (int): Maximum number of ingredients to return
            
        Returns:
            list[str]: Detected ingredient names
        """
        threshold = threshold if threshold is not None else THRESHOLD
        top_k = top_k if top_k is not None else TOP_K
        
        image = Image.open(image_path).convert("RGB")
        return self.recognize(image, threshold, top_k)
    
    def recognize(self, image, threshold=None, top_k=None):
        """
        Recognize food ingredients from a PIL Image.
        
        Args:
            image (PIL.Image): Input image
            threshold (float): Confidence threshold for ingredient detection
            top_k (int): Maximum number of ingredients to return
            
        Returns:
            list[str]: Detected ingredient names
        """
        threshold = threshold if threshold is not None else THRESHOLD
        top_k = top_k if top_k is not None else TOP_K
        
        texts = [f"a photo of {ingredient}" for ingredient in self.ingredients]
        
        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        detected_ingredients = []
        for ingredient, prob in zip(self.ingredients, probs):
            if prob >= threshold:
                detected_ingredients.append({
                    "ingredient": ingredient,
                    "confidence": float(prob)
                })


        detected_ingredients = sorted(detected_ingredients, 
                                      key=lambda x: x["confidence"], 
                                      reverse=True)[:top_k]
        
        
        return [item["ingredient"] for item in detected_ingredients]
    
    def recognize_batch(self, images, threshold=None, top_k=None):
        """
        Process multiple images in batch.
        """
        threshold = threshold if threshold is not None else THRESHOLD
        top_k = top_k if top_k is not None else TOP_K
        
        results = []
        for image in images:
            results.append(self.recognize(image, threshold, top_k))
        return results


def set_threshold(value):
    global THRESHOLD
    THRESHOLD = value
    
def set_top_k(value):
    global TOP_K
    TOP_K = value

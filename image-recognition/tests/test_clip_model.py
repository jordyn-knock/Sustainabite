import os
import argparse
import sys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
sys.path.append('/home/alissah/youCode/youcode/image-recognition')
from clip_model import FoodRecognizer


def display_results(image_path, ingredients, save_output=True, output_path="output_visualization.jpg"):
    """
    Display the image with detected ingredients.
    
    Args:
        image_path (str): Path to the input image
        ingredients (list): List of detected ingredients
        save_output (bool): Whether to save the output visualization
        output_path (str): Path to save the output image
    """
    img = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(12, 10))
    
    plt.imshow(img)
    plt.axis('off')
    
    ingredients_text = "\n".join([f"• {ing}" for ing in ingredients])
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.text(img.width * 0.05, img.height * 0.05, 
             f"Detected Ingredients:\n{ingredients_text}", 
             fontsize=12, verticalalignment='top', bbox=props)
    
    plt.title("Food Recognition Results", fontsize=15)
    
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight')
    print(f"Output saved to {output_path}")
    
    # plt.show()  # Comment out or remove this line

def main():
    parser = argparse.ArgumentParser(description='Test CLIP Food Recognition')
    parser.add_argument('--image', type=str, required=True, help='Path to the food image')
    parser.add_argument('--threshold', type=float, default=0.2, help='Confidence threshold')
    parser.add_argument('--top_k', type=int, default=10, help='Maximum number of ingredients to return')
    parser.add_argument('--save_output', action='store_true', help='Save visualization')
    parser.add_argument('--output_path', type=str, default='output.jpg', help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Initialize the recognizer
    print("Initializing CLIP model...")
    recognizer = FoodRecognizer()
    
    # Recognize ingredients
    print(f"Processing image: {args.image}")
    ingredients = recognizer.recognize_from_file(
        args.image, 
        threshold=args.threshold, 
        top_k=args.top_k
    )
    
    # Print results
    print("\nDetected Ingredients:")
    for i, ingredient in enumerate(ingredients):
        print(f"{i+1}. {ingredient}")
    
    # Display results
    display_results(
        args.image, 
        ingredients, 
        save_output=args.save_output, 
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()
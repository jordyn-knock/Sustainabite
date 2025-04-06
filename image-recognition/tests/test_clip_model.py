import os
import argparse
import sys
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

sys.path.append(PARENT_DIR)

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
    
    # Show the detected ingredients in a text box
    ingredients_text = "\n".join([f"• {ing}" for ing in ingredients])
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.text(img.width * 0.05, img.height * 0.05, 
             f"Detected Ingredients:\n{ingredients_text}", 
             fontsize=12, verticalalignment='top', bbox=props)
    
    plt.title("Food Recognition Results", fontsize=15)
    plt.tight_layout()

    if save_output:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Output saved to {output_path}")
    else:
        print("Not saving output image.")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test CLIP Food Recognition')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the food image (relative or absolute)')
    parser.add_argument('--save_output', action='store_true',
                        help='Whether to save the output visualization')
    parser.add_argument('--output_path', type=str, default='output_visualization.jpg',
                        help='Path to save the visualization image')
    args = parser.parse_args()

    top_ingredients_csv = os.path.join(PARENT_DIR, "data", "top_500_ingredients.csv")

    df = pd.read_csv(top_ingredients_csv)  
    top_ingredients = df["ingredient"].tolist()
    
    recognizer = FoodRecognizer()  
    print("Default ingredient count:", len(recognizer.ingredients))
    
    recognizer.ingredients = top_ingredients
    print("New ingredient count:", len(recognizer.ingredients))
    
    detected = recognizer.recognize_from_file(args.image)
    print("\nDetected ingredients:")
    for ing in detected:
        print(f"- {ing}")

    display_results(
        image_path=args.image,
        ingredients=detected,
        save_output=args.save_output,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()

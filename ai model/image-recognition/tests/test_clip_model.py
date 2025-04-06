import os
import argparse
import sys
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Get directory paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PARENT_DIR)
print(f"Current directory: {CURRENT_DIR}")
print(f"Parent directory: {PARENT_DIR}")

# Import the FoodRecognizer class
from clip_model import FoodRecognizer


def display_results(image_path, ingredients_probs, save_output=True, output_path="output_visualization.jpg"):
    """
    Display the image with detected ingredients and probabilities.

    Args:
        image_path (str): Path to the input image
        ingredients_probs (list): List of tuples (ingredient, probability)
        save_output (bool): Whether to save the output visualization
        output_path (str): Path to save the output image
    """
    try:
        print(f"Opening image for display: {image_path}")
        img = Image.open(image_path).convert("RGB")
        plt.figure(figsize=(12, 10))

        plt.imshow(img)
        plt.axis('off')

        # Show the detected ingredients with probabilities in a text box
        ingredients_text = "\n".join([f"• {ing}: {prob:.2%}" for ing, prob in ingredients_probs])
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        plt.text(img.width * 0.05, img.height * 0.05,
                f"Detected Ingredients:\n{ingredients_text}",
                fontsize=12, verticalalignment='top', bbox=props)

        plt.title("Food Recognition Results", fontsize=15)
        plt.tight_layout()

        if save_output:
            # Make sure the output directory exists
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
            
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Output saved to {output_path}")
        else:
            print("Not saving output image.")

        plt.close()
    except Exception as e:
        print(f"Error in display_results: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Test CLIP Food Recognition')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the food image (relative or absolute)')
    parser.add_argument('--save_output', action='store_true',
                        help='Whether to save the output visualization')
    parser.add_argument('--output_path', type=str, default='output_visualization.jpg',
                        help='Path to save the visualization image')
    args = parser.parse_args()

    # Handle both absolute and relative paths for the image
    image_path = args.image
    if not os.path.isabs(image_path):
        # If relative path, make it relative to the current directory
        image_path = os.path.abspath(os.path.join(CURRENT_DIR, image_path))
    
    print(f"Using image path: {image_path}")
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist at {image_path}")
        print(f"Current working directory: {os.getcwd()}")
        sys.exit(1)

    try:
        # Load top ingredients from CSV
        top_ingredients_csv = os.path.join(PARENT_DIR, "data", "top_500_ingredients.csv")
        print(f"Looking for ingredients CSV at: {top_ingredients_csv}")
        
        if os.path.exists(top_ingredients_csv):
            df = pd.read_csv(top_ingredients_csv)
            top_ingredients = df["ingredient"].tolist()
            print(f"Loaded {len(top_ingredients)} ingredients from CSV")
        else:
            print(f"Warning: Ingredients CSV not found at {top_ingredients_csv}")
            top_ingredients = []

        # Initialize recognizer
        recognizer = FoodRecognizer(ingredients_csv=top_ingredients_csv)
        
        # Make sure we have ingredients
        if top_ingredients:
            recognizer.ingredients = top_ingredients

        # Run recognition
        detected_probs = recognizer.recognize_from_file(image_path)

        print("\nDetected ingredients with probabilities:")
        for ing, prob in detected_probs:
            print(f"- {ing}: {prob:.2%}")

        # Display results
        display_results(
            image_path=image_path,
            ingredients_probs=detected_probs,
            save_output=args.save_output,
            output_path=args.output_path
        )
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
import pandas as pd
import ast

INPUT_CSV = "data/recipes_ingredients.csv"
OUTPUT_CSV = "data/top_500_ingredients.csv"
TOP_K = 500  

def main():
    df = pd.read_csv(INPUT_CSV, usecols=["ingredients"])
    
    ingredient_counts = {}
    
    for row in df["ingredients"]:
        try:
            ing_list = ast.literal_eval(row)
            if not isinstance(ing_list, list):
                continue
        except (SyntaxError, ValueError):
            continue
        
        for ing in ing_list:
            ing = ing.strip().lower()
            ingredient_counts[ing] = ingredient_counts.get(ing, 0) + 1
    
    sorted_ing = sorted(ingredient_counts.items(), key=lambda x: x[1], reverse=True)
    
    top_ingredients = sorted_ing[:TOP_K]
    
    df_top = pd.DataFrame(top_ingredients, columns=["ingredient", "count"])
    
    df_top.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved the top {TOP_K} ingredients to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

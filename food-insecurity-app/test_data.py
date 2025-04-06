TEST_IMAGES = {
    # 1) pasta-ingredients.jpg
    "pasta-ingredients.jpg": {
        "ingredients": [
            "Uncooked pasta", "Tomatoes", "Onion", "Garlic",
            "Basil", "Olive oil", "Shredded cheese"
        ],
        "recipes": [
            {
                "title": "Minimal‐Waste Marinara Pasta",
                "image_url": "https://www.kimscravings.com/wp-content/uploads/2022/12/creamy-pasta-sauce.jpg",
                "time": "30 minutes",
                "ingredients": [
                    {"name": "pasta", "amount": "8 oz"},
                    {"name": "tomatoes", "amount": "2 cups, diced"},
                    {"name": "onion", "amount": "1 small, chopped"},
                    {"name": "garlic", "amount": "2 cloves, minced"},
                    {"name": "basil", "amount": "handful, fresh"},
                    {"name": "olive oil", "amount": "1 tbsp"}
                ],
                "steps": [
                    "Boil water and cook pasta until al dente.",
                    "Sauté onion and garlic in olive oil.",
                    "Add tomatoes and simmer for 10–15 minutes.",
                    "Stir in basil, then toss with drained pasta.",
                    "Top with shredded cheese (optional)."
                ]
            },
            {
                "title": "One‐Pan Pasta Bake",
                "image_url": "https://www.halfbakedharvest.com/wp-content/uploads/2020/11/One-Pan-4-Cheese-Sun-Dried-Tomato-and-Spinach-Pasta-Bake-1.jpg",
                "time": "35 minutes",
                "ingredients": [
                    {"name": "pasta", "amount": "8 oz"},
                    {"name": "tomatoes", "amount": "1 can crushed"},
                    {"name": "onion", "amount": "1, diced"},
                    {"name": "garlic", "amount": "2 cloves, crushed"},
                    {"name": "cheese", "amount": "1 cup, shredded"},
                    {"name": "basil", "amount": "handful, chopped"}
                ],
                "steps": [
                    "Preheat oven to 375°F (190°C).",
                    "In a baking dish, mix pasta with crushed tomatoes, onion, and garlic.",
                    "Add a splash of water or stock, then cover and bake ~20 minutes.",
                    "Uncover, sprinkle cheese on top, bake 5 more minutes.",
                    "Garnish with fresh basil."
                ]
            },
            {
                "title": "Tomato & Basil Pasta Salad",
                "image_url": "https://www.eatingwell.com/thmb/vuV5Gs5aB-P7B84OJTmF_9lDlrE=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/Easy-Tomato-Pasta-Salad-2000-d4fccc6f46764ef2ab0a741e3ee1f9fa.jpg",
                "time": "20 minutes",
                "ingredients": [
                    {"name": "pasta", "amount": "8 oz (any shape)"},
                    {"name": "tomatoes", "amount": "1 cup, diced"},
                    {"name": "basil", "amount": "handful, chopped"},
                    {"name": "olive oil", "amount": "2 tbsp"},
                    {"name": "onion", "amount": "1/4 cup, finely chopped"},
                    {"name": "garlic", "amount": "1 clove, minced"}
                ],
                "steps": [
                    "Cook pasta per package instructions, then rinse with cold water.",
                    "Combine tomatoes, onion, garlic, and basil in a bowl.",
                    "Toss in cooled pasta, drizzle with olive oil.",
                    "Season with salt and pepper to taste."
                ]
            },
            {
                "title": "Creamy Tomato Sauce with Garlic",
                "image_url": "https://majasrecipes.com/wp-content/uploads/2024/09/tomato-garlic-pasta-1.jpg",
                "time": "25 minutes",
                "ingredients": [
                    {"name": "tomatoes", "amount": "2 cups, chopped"},
                    {"name": "garlic", "amount": "2 cloves, minced"},
                    {"name": "heavy cream", "amount": "1/2 cup"},
                    {"name": "onion", "amount": "1/2, diced"},
                    {"name": "olive oil", "amount": "1 tbsp"},
                    {"name": "pasta", "amount": "8 oz"}
                ],
                "steps": [
                    "Cook pasta in salted water until al dente.",
                    "Sauté onion and garlic in olive oil.",
                    "Add tomatoes and simmer until soft.",
                    "Stir in heavy cream, reduce heat slightly.",
                    "Combine sauce with cooked pasta."
                ]
            },
            {
                "title": "Pasta alla Norma",
                "image_url": "https://cookieandkate.com/images/2020/09/pasta-alla-norma-recipe-3.jpg",
                "time": "40 minutes",
                "ingredients": [
                    {"name": "pasta", "amount": "8 oz"},
                    {"name": "eggplant", "amount": "1 medium, diced"},
                    {"name": "tomatoes", "amount": "1 can, crushed"},
                    {"name": "onion", "amount": "1, sliced"},
                    {"name": "garlic", "amount": "2 cloves, sliced"},
                    {"name": "cheese", "amount": "ricotta salata or parm, optional"}
                ],
                "steps": [
                    "Salt eggplant and let drain for 15 min, pat dry.",
                    "Fry or roast eggplant until golden.",
                    "Sauté onion and garlic, add crushed tomatoes, simmer.",
                    "Add eggplant, then stir in cooked pasta.",
                    "Top with grated cheese if desired."
                ]
            }
        ]
    },

    # 2) test_fridge_1.jpeg
    "test_fridge_1.jpeg": {
        "ingredients": [
            "Milk", "Eggs", "Spinach", "Mushrooms",
            "Leftover chicken", "Cheese"
        ],
        "recipes": [
            # Minimal placeholders or expand similarly
            {
                "title": "Spinach & Mushroom Egg Scramble",
                "image_url": "https://www.skinnytaste.com/wp-content/uploads/2022/01/Mushroom-Spinach-Scrambled-Eggs-6.jpg",
                "time": "15 minutes",
                "ingredients": [{"name": "eggs", "amount": "3"},
                                {"name": "milk", "amount": "1/4 cup"},
                                {"name": "salt", "amount": "a pinch"},
                                {"name": "pepper", "amount": "a pinch"},
                                {"name": "spinach", "amount": "1 cup"},
                                {"name": "mushrooms", "amount": "1/2 cup"},
                                {"name": "cheese", "amount": "2 tbsp"}],
                "steps": [
                    "Crack eggs into a bowl, add salt, pepper, and milk (optional), then whisk.",
                    "Heat oil or butter (or some other substitute)",
                    "Add sliced mushrooms and cook until soft and golden.",
                    "Add garlic (optional), then chopped spinach and cook until wilted.",
                    "Pour in eggs and gently stir to scramble.",
                    "Add cheese just before eggs are fully set.",
                    "Remove from heat and serve warm."
                ]
            },
            {
                "title": "Leftover Chicken & Veggie Omelet",
                "image_url": "https://khinskitchen.com/wp-content/uploads/2023/04/chicken-omelette-04.jpg", #do this 
                "time": "15 minutes",
                "ingredients": [{ "name": "eggs", "amount": "3" },
                                { "name": "milk", "amount": "1/4 cup" },
                                { "name": "salt", "amount": "a pinch" },
                                { "name": "pepper", "amount": "a pinch" },
                                { "name": "leftover cooked chicken", "amount": "1/2 cup, chopped" },
                                { "name": "leftover cooked vegetables", "amount": "1/2 cup, chopped" },
                                { "name": "cheese", "amount": "2 tbsp (optional)" }],
                "steps": [
                    "Crack eggs into a bowl, add salt, pepper, and milk (optional), then whisk.",
                    "Heat oil or butter in a pan over medium heat.",
                    "Add chopped leftover chicken and vegetables, cook until heated through.",
                    "Pour eggs over the mixture, tilting pan to spread evenly.",
                    "Cook until edges are set, then gently lift sides to let uncooked egg flow underneath.",
                    "Sprinkle cheese on top (optional), then fold omelet in half.",
                    "Cook for another minute, then slide onto a plate and serve."
                ]
            },
            {
                "title": "Creamy Mushroom Soup",
                "image_url": "https://rainbowplantlife.com/wp-content/uploads/2022/11/Mushroom-soup-cover-image-1-of-1.jpg",
                "time": "25 minutes",
                "ingredients": [{ "name": "mushrooms", "amount": "2 cups, sliced" },
                { "name": "onion", "amount": "1 small, chopped" },
                { "name": "garlic", "amount": "2 cloves, minced" },
                { "name": "butter or oil", "amount": "1 tbsp" },
                { "name": "broth or water", "amount": "2 cups" },
                { "name": "milk", "amount": "1 cup" },
                { "name": "salt", "amount": "to taste" },
                { "name": "pepper", "amount": "to taste" }],
                "steps": ["Heat butter or oil in a pot over medium heat.",
                    "Add chopped onion and cook until soft.",
                    "Add garlic and sliced mushrooms, cook until mushrooms are browned and tender.",
                    "Pour in broth and bring to a boil, then reduce heat and simmer for 10 minutes.",
                    "Use a blender or immersion blender to blend the soup until smooth (optional).",
                    "Stir in milk or cream, then season with salt and pepper.",
                    "Simmer for 2-3 more minutes, then serve warm."
                ]
            },
            {
                "title": "Spinach‐Mushroom Quesadillas",
                "image_url": "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.budgetbytes.com%2Fspinach-and-mushroom-quesadillas%2F&psig=AOvVaw0DLjIpBT0gxGgQFW3voT_L&ust=1744051566153000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCOCsgcKIxIwDFQAAAAAdAAAAABAE",
                "time": "20 minutes",
                "ingredients": [{ "name": "tortillas", "amount": "2 large" },
                                { "name": "spinach", "amount": "1 cup" },
                                { "name": "mushrooms", "amount": "1/2 cup" },
                                { "name": "cheese", "amount": "1/2 cup" },
                                { "name": "butter or oil", "amount": "1 tbsp" },
                                { "name": "garlic", "amount": "1 clove, minced (optional)" },
                                { "name": "salt", "amount": "to taste" },
                                { "name": "pepper", "amount": "to taste" }],
                "steps": ["Heat butter or oil in a pan over medium heat.",
                    "Add garlic (optional) and sliced mushrooms, cook until mushrooms are soft.",
                    "Add chopped spinach, cook until wilted, then season with salt and pepper.",
                    "Remove mixture from pan and set aside.",
                    "Place one tortilla in the pan, sprinkle half the cheese on one side.",
                    "Add spinach-mushroom mix, then top with remaining cheese and fold tortilla in half.",
                    "Cook until golden brown on both sides and cheese is melted.",
                    "Cut into wedges and serve warm."
                ]
            },
            {
                "title": "Chicken Alfredo Bake",
                "image_url": "https://www.allrecipes.com/thmb/0-CTkaneK0crlqqaIsEtG0oojtk=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/276726-ChickenAlfredoCasserole-mfs-2X3-0070-6389ebe9a83642d9ba55ac30fd2d8683.jpg",
                "time": "30 minutes",
                "ingredients": [{ "name": "cooked pasta", "amount": "3 cups (penne or fusilli)" },
                { "name": "cooked chicken", "amount": "2 cups, chopped or shredded" },
                { "name": "alfredo sauce", "amount": "2 cups (store-bought or homemade)" },
                { "name": "cheese", "amount": "1 cup, shredded (mozzarella or a mix)" },
                { "name": "salt", "amount": "to taste" },
                { "name": "pepper", "amount": "to taste" },
                { "name": "parsley", "amount": "for garnish (optional)" }],
                "steps": ["Preheat oven to 375°F (190°C).",
                    "In a large bowl, mix cooked pasta, chicken, and Alfredo sauce.",
                    "Add salt and pepper to taste, and stir until evenly coated.",
                    "Transfer mixture to a baking dish.",
                    "Sprinkle shredded cheese evenly on top.",
                    "Bake for 20–25 minutes, until cheese is melted and bubbly.",
                    "Garnish with parsley (optional) and serve warm."
                ]
            }
        ]
    },

    # 3) test_fridge_2.jpeg
    "test_fridge_2.jpeg": {
        "ingredients": [
            "Bell peppers", "Carrots", "Lettuce",
            "Deli meat", "Yogurt", "Cheddar cheese"
        ],
        "recipes": [
            {
                "title": "No‐Waste Lettuce Wraps",
                "image_url": "https://www.gettystewart.com/wp-content/uploads/2021/11/chicken-lettuce-wrap-close-on-board-sq2.jpg",
                "time": "10 minutes",
                "ingredients": [{ "name": "lettuce", "amount": "6 large leaves (romaine or iceberg)" },
                                { "name": "bell peppers", "amount": "1/2 cup, thinly sliced" },
                                { "name": "carrots", "amount": "1/2 cup, shredded or julienned" },
                                { "name": "deli meat", "amount": "6 slices (turkey, ham, or your choice)" },
                                { "name": "yogurt", "amount": "1/4 cup (plain or Greek, for spread)" },
                                { "name": "cheddar cheese", "amount": "1/2 cup, shredded or sliced" },
                                { "name": "salt", "amount": "to taste (optional)" },
                                { "name": "pepper", "amount": "to taste (optional)" }],
                "steps": ["Lay out clean, dry lettuce leaves on a flat surface.",
                    "Spread a thin layer of yogurt on each leaf as a light dressing.",
                    "Place a slice of deli meat on each leaf.",
                    "Add sliced bell peppers, shredded carrots, and cheddar cheese on top.",
                    "Sprinkle with salt and pepper if desired.",
                    "Roll or fold the lettuce leaves into wraps.",
                    "Serve immediately, or secure with toothpicks for a snack or lunchbox option."
                ]

            },
            {
                "title": "Chef’s Salad",
                "image_url": "https://www.spendwithpennies.com/wp-content/uploads/2024/04/Chef-Salad-SpendWithPennies-2.jpg",
                "time": "15 minutes",
                "ingredients": [{ "name": "lettuce", "amount": "2 cups, chopped" },
                                { "name": "bell peppers", "amount": "1/2 cup, sliced" },
                                { "name": "carrots", "amount": "1/2 cup, shredded" },
                                { "name": "deli meat", "amount": "3 slices, chopped (turkey, ham, etc.)" },
                                { "name": "cheddar cheese", "amount": "1/2 cup, shredded or cubed" },
                                { "name": "yogurt", "amount": "1/4 cup (as a dressing base)" },
                                { "name": "salt", "amount": "to taste (optional)" },
                                { "name": "pepper", "amount": "to taste (optional)" }],
                "steps": ["Add chopped lettuce to a large salad bowl.",
                            "Top with sliced bell peppers and shredded carrots.",
                            "Add chopped deli meat and cheddar cheese.",
                            "In a small bowl, mix yogurt with a pinch of salt and pepper for dressing.",
                            "Drizzle yogurt dressing over the salad.",
                            "Toss gently to combine and serve immediately."
                        ]
            },
            {
                "title": "Bell Pepper & Carrot Stir‐Fry",
                "image_url": "https://thewoksoflife.com/wp-content/uploads/2022/02/vegetable-stir-fry-9.jpg",
                "time": "20 minutes",
                "ingredients": [{ "name": "bell peppers", "amount": "1 cup, sliced" },
                                { "name": "carrots", "amount": "1 cup, julienned or thinly sliced" },
                                { "name": "garlic", "amount": "2 cloves, minced (optional)" },
                                { "name": "oil", "amount": "1 tbsp (vegetable or olive oil)" },
                                { "name": "soy sauce", "amount": "2 tbsp" },
                                { "name": "salt", "amount": "to taste" },
                                { "name": "pepper", "amount": "to taste" },
                                { "name": "sesame seeds", "amount": "1 tsp (optional for garnish)" }],
                "steps": ["Heat oil in a pan or wok over medium-high heat.",
                            "Add garlic (optional) and sauté for 30 seconds until fragrant.",
                            "Add sliced bell peppers and carrots to the pan.",
                            "Stir-fry for 5–7 minutes until vegetables are tender-crisp.",
                            "Add soy sauce, salt, and pepper, then toss to coat evenly.",
                            "Cook for another minute, then remove from heat.",
                            "Garnish with sesame seeds (optional) and serve warm."
                        ]
            },
            {
                "title": "Yogurt Veggie Dip",
                "image_url": "https://www.fivehearthome.com/wp-content/uploads/2022/09/Greek-Yogurt-Dip-Recipe-by-Five-Heart-Home_1000pxFeatured60.jpg",
                "time": "5 minutes",
                "ingredients": [{ "name": "yogurt", "amount": "1 cup (plain or Greek)" },
                                { "name": "carrots", "amount": "1/4 cup, finely grated or chopped" },
                                { "name": "bell peppers", "amount": "1/4 cup, finely chopped" },
                                { "name": "garlic", "amount": "1 clove, minced (optional)" },
                                { "name": "salt", "amount": "to taste" },
                                { "name": "pepper", "amount": "to taste" },
                                { "name": "lemon juice", "amount": "1 tsp (optional)" },
                                { "name": "herbs", "amount": "1 tsp (dried or fresh parsley, dill, or chives – optional)" }],
                "steps": ["Add yogurt to a mixing bowl.",
                            "Mix in finely chopped carrots and bell peppers.",
                            "Add garlic, lemon juice, herbs, salt, and pepper.",
                            "Stir everything together until well combined.",
                            "Chill in the fridge for 10–15 minutes for best flavor.",
                            "Serve with fresh veggies, crackers, or as a spread."
                        ]
            },
            {
                "title": "Veggie‐Stuffed Quesadillas",
                "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQUdYOmsN1Z88ABEHzSx7KfQnzd1mL5knW0XA&s",
                "time": "15 minutes",
                "ingredients": [{ "name": "tortillas", "amount": "2 large" },
                                { "name": "bell peppers", "amount": "1/2 cup, sliced" },
                                { "name": "carrots", "amount": "1/4 cup, shredded or julienned" },
                                { "name": "spinach", "amount": "1 cup, chopped" },
                                { "name": "cheddar cheese", "amount": "1/2 cup, shredded" },
                                { "name": "oil or butter", "amount": "1 tbsp" },
                                { "name": "salt", "amount": "to taste" },
                                { "name": "pepper", "amount": "to taste" }],
                "steps": ["Heat oil or butter in a pan over medium heat.",
                            "Add bell peppers, carrots, and spinach, cook until softened.",
                            "Season with salt and pepper, then remove from heat.",
                            "Place one tortilla in a clean pan or skillet.",
                            "Spread half the cheese on one side of the tortilla.",
                            "Add cooked veggies, then top with the rest of the cheese.",
                            "Fold tortilla in half and cook until golden on both sides and cheese is melted.",
                            "Cut into wedges and serve warm."
                        ]
            }
        ]
    },

    # 4) test_grocery_haul_0.jpg
    "test_grocery_haul_0.jpg": {
        "ingredients": [
            "Chicken breasts", "Broccoli florets", "Potatoes",
            "Cherry tomatoes", "Cilantro", "Onions"
        ],
        "recipes": [
            {
                "title": "Sheet Pan Chicken & Veggies",
                "image_url": "https://www.tasteofhome.com/wp-content/uploads/2018/01/Pan-Roasted-Chicken-and-Vegetables_EXPS_LECBZ23_134862_P2_MD_09_07_1b_v1.jpg",
                "time": "30 minutes",
                "ingredients": [{ "name": "chicken breasts", "amount": "2, cut into chunks or left whole" },
                                { "name": "broccoli florets", "amount": "2 cups" },
                                { "name": "potatoes", "amount": "2 medium, diced" },
                                { "name": "cherry tomatoes", "amount": "1 cup" },
                                { "name": "onions", "amount": "1 large, sliced" },
                                { "name": "cilantro", "amount": "2 tbsp, chopped (for garnish)" },
                                { "name": "oil", "amount": "2 tbsp (olive or vegetable)" },
                                { "name": "salt", "amount": "to taste" },
                                { "name": "pepper", "amount": "to taste" },
                                { "name": "garlic powder", "amount": "1 tsp (optional)" },
                                { "name": "paprika", "amount": "1 tsp (optional)" }],
                "steps": ["Preheat oven to 400°F (200°C).",
                            "Add chicken, potatoes, broccoli, cherry tomatoes, and onions to a large bowl.",
                            "Drizzle with oil, then season with salt, pepper, garlic powder, and paprika (if using).",
                            "Toss everything until evenly coated.",
                            "Spread evenly on a large sheet pan in a single layer.",
                            "Bake for 25–30 minutes, or until chicken is cooked through and veggies are tender.",
                            "Garnish with chopped cilantro and serve warm."
                        ]
            },
            {
                "title": "Chicken & Broccoli Stir‐Fry",
                "image_url": "https://www.allrecipes.com/thmb/wbWfb-cXCq1yd6NlpJR7z5Nr1Wc=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/240708-broccoli-and-chicken-stir-fry-DDMFS-4x3-27886203cb2744f99d15247496019942.jpg",
                "time": "20 minutes",
                "ingredients": [{ "name": "chicken breasts", "amount": "2, sliced into thin strips" },
                                { "name": "broccoli florets", "amount": "2 cups" },
                                { "name": "garlic", "amount": "2 cloves, minced" },
                                { "name": "soy sauce", "amount": "3 tbsp" },
                                { "name": "oil", "amount": "1 tbsp (vegetable or sesame oil)" },
                                { "name": "salt", "amount": "to taste" },
                                { "name": "pepper", "amount": "to taste" },
                                { "name": "cornstarch", "amount": "1 tsp (optional, for thickening)" },
                                { "name": "water", "amount": "1/4 cup (if using cornstarch)" }],
                "steps": ["Heat oil in a pan or wok over medium-high heat.",
                            "Add chicken strips, season with salt and pepper, and cook until no longer pink.",
                            "Remove chicken and set aside.",
                            "In the same pan, add broccoli and a splash of water, cook until just tender.",
                            "Add garlic and cook for 30 seconds until fragrant.",
                            "Return chicken to the pan, add soy sauce, and stir to combine.",
                            "Optional: Mix cornstarch with water and add to the pan to thicken sauce.",
                            "Cook for 1–2 more minutes until everything is well coated and heated through.",
                            "Serve hot with rice or noodles."
                        ]
            },
            {
                "title": "Cilantro‐Lime Chicken",
                "image_url": "",
                "time": "25 minutes",
                "ingredients": [],
                "steps": []
            },
            {
                "title": "Oven‐Roasted Chicken with Potatoes",
                "image_url": "",
                "time": "35 minutes",
                "ingredients": [],
                "steps": []
            },
            {
                "title": "Tomato & Cilantro Chicken Soup",
                "image_url": "",
                "time": "40 minutes",
                "ingredients": [],
                "steps": []
            }
        ]
    },

    # 5) test_grocery_haul_1.jpg
    "test_grocery_haul_1.jpg": {
    "ingredients": [
        "Ground beef", "Tortillas", "Avocado",
        "Onion", "Bell peppers", "Cheese", "Lettuce"
    ],
    "recipes": [
        {
        "title": "Leftover‐Friendly Tacos",
        "image_url": "https://wholesomemadeeasy.com/wp-content/uploads/2021/12/Rotisserie-Chicken-Tacos-22.jpg",
        "time": "15 minutes",
        "ingredients": [
            { "name": "tortillas", "amount": "4" },
            { "name": "ground beef", "amount": "1/2 lb" },
            { "name": "onion", "amount": "1/2, chopped" },
            { "name": "cheese", "amount": "1/2 cup, shredded" },
            { "name": "lettuce", "amount": "1 cup, shredded" },
            { "name": "avocado", "amount": "1, sliced" }
        ],
        "steps": [
            "Cook ground beef and chopped onion in a pan until browned.",
            "Warm tortillas in a pan or microwave.",
            "Layer beef, cheese, lettuce, and avocado onto tortillas.",
            "Fold or roll and serve warm."
        ]
        },
        {
        "title": "Bell Pepper & Beef Skillet",
        "image_url": "https://juliasalbum.com/wp-content/uploads/2023/09/Ground-Beef-Stir-Fry-1.jpg",
        "time": "20 minutes",
        "ingredients": [
            { "name": "ground beef", "amount": "1/2 lb" },
            { "name": "bell peppers", "amount": "1 cup, sliced" },
            { "name": "onion", "amount": "1/2, chopped" },
            { "name": "cheese", "amount": "1/4 cup, optional" }
        ],
        "steps": [
            "Sauté ground beef in a skillet until mostly cooked.",
            "Add bell peppers and onions, cook until tender.",
            "Top with cheese (optional) and serve hot."
        ]
        },
        {
        "title": "Avocado Lettuce Wrap Burgers",
        "image_url": "https://images.getrecipekit.com/20230316014437-avocdaoburgerlettucewraps_0340_2.jpg?aspect_ratio=16:9&quality=90&",
        "time": "25 minutes",
        "ingredients": [
            { "name": "ground beef", "amount": "1/2 lb" },
            { "name": "lettuce", "amount": "6 large leaves" },
            { "name": "avocado", "amount": "1, sliced" },
            { "name": "cheese", "amount": "1/4 cup, shredded" },
            { "name": "onion", "amount": "1/4, thinly sliced" }
        ],
        "steps": [
            "Form beef into small patties and cook until browned and cooked through.",
            "Place patties onto lettuce leaves.",
            "Top with avocado, cheese, and onion.",
            "Wrap and serve."
        ]
        },
        {
        "title": "Stuffed Bell Peppers",
        "image_url": "https://tyberrymuch.com/wp-content/uploads/2020/09/vegan-stuffed-peppers-recipe-720x720.jpg",
        "time": "35 minutes",
        "ingredients": [
            { "name": "bell peppers", "amount": "2, halved and hollowed" },
            { "name": "ground beef", "amount": "1/2 lb" },
            { "name": "onion", "amount": "1/2, chopped" },
            { "name": "cheese", "amount": "1/4 cup, shredded" }
        ],
        "steps": [
            "Preheat oven to 375°F (190°C).",
            "Cook beef with onion in a pan until browned.",
            "Stuff the beef mixture into bell pepper halves.",
            "Top with cheese and bake for 20–25 minutes."
        ]
        },
        {
        "title": "Beef & Avocado Nachos",
        "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT6CrFn-xfklgPZZsP_QHkZ4zPAKHjbGZ6TXw&s",
        "time": "20 minutes",
        "ingredients": [
            { "name": "tortillas", "amount": "cut into chips or use store-bought" },
            { "name": "ground beef", "amount": "1/2 lb" },
            { "name": "cheese", "amount": "1/2 cup, shredded" },
            { "name": "avocado", "amount": "1, diced" },
            { "name": "onion", "amount": "1/4, chopped" }
        ],
        "steps": [
            "Preheat oven to 375°F (190°C).",
            "Cook ground beef until browned.",
            "Spread chips on a baking sheet, top with beef and cheese.",
            "Bake for 10 minutes, then add avocado and onion before serving."
        ]
        }
    ]
    },


    # 6) test_tacos.jpeg
    "test_tacos.jpeg": {
    "ingredients": [
        "Tortillas", "Seasoned ground beef/pork", "Shredded cheese",
        "Tomatoes or salsa", "Lettuce", "Onions", "Lime"
    ],
    "recipes": [
        {
        "title": "Leftover Taco Salad",
        "image_url": "https://www.hotpankitchen.com/wp-content/uploads/2024/02/leftover-steak-taco-salad-2.jpg",
        "time": "10 minutes",
        "ingredients": [
            { "name": "lettuce", "amount": "2 cups, chopped" },
            { "name": "seasoned meat", "amount": "1/2 cup" },
            { "name": "cheese", "amount": "1/4 cup, shredded" },
            { "name": "tomatoes or salsa", "amount": "1/4 cup" },
            { "name": "onions", "amount": "2 tbsp, chopped" },
            { "name": "lime", "amount": "wedges for serving" }
        ],
        "steps": [
            "Add chopped lettuce to a bowl.",
            "Top with leftover taco meat, cheese, tomatoes or salsa, and onions.",
            "Squeeze lime over the top and serve."
        ]
        },
        {
        "title": "https://www.maricruzavalos.com/wp-content/uploads/2021/07/taco_quesadillas_recipe.jpg",
        "image_url": "",
        "time": "15 minutes",
        "ingredients": [
            { "name": "tortillas", "amount": "2" },
            { "name": "seasoned meat", "amount": "1/2 cup" },
            { "name": "cheese", "amount": "1/2 cup, shredded" },
            { "name": "onion", "amount": "2 tbsp, chopped" }
        ],
        "steps": [
            "Place one tortilla in a pan, sprinkle with cheese, meat, and onions.",
            "Top with second tortilla and cook until golden on both sides.",
            "Slice and serve."
        ]
        },
        {
        "title": "Taco‐Stuffed Bell Peppers",
        "image_url": "https://www.paleorunningmomma.com/wp-content/uploads/2021/07/taco-stuffed-peppers-3.jpg",
        "time": "25 minutes",
        "ingredients": [
            { "name": "bell peppers", "amount": "2, halved and hollowed" },
            { "name": "seasoned meat", "amount": "1/2 cup" },
            { "name": "cheese", "amount": "1/4 cup, shredded" },
            { "name": "tomatoes or salsa", "amount": "2 tbsp" }
        ],
        "steps": [
            "Preheat oven to 375°F (190°C).",
            "Stuff bell pepper halves with meat, cheese, and salsa.",
            "Bake for 20 minutes until peppers are tender and cheese is melted."
        ]
        },
        {
        "title": "Taco Pizza",
        "image_url": "https://www.allrecipes.com/thmb/RxreNop4KcxV0GGgb2zAEZo_D6c=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/44410taco-pizzafabeveryday4x3-01863597aa224c32b35b970b5b744b20.jpg",
        "time": "30 minutes",
        "ingredients": [
            { "name": "tortilla", "amount": "1 large (as base)" },
            { "name": "seasoned meat", "amount": "1/3 cup" },
            { "name": "cheese", "amount": "1/3 cup, shredded" },
            { "name": "tomatoes or salsa", "amount": "2 tbsp" },
            { "name": "lettuce", "amount": "1/2 cup, shredded" },
            { "name": "onion", "amount": "2 tbsp, chopped" }
        ],
        "steps": [
            "Preheat oven to 400°F (200°C).",
            "Place tortilla on a baking tray, spread with meat and cheese.",
            "Bake for 8–10 minutes until crispy.",
            "Top with lettuce, onion, and salsa before serving."
        ]
        },
        {
        "title": "Fully‐Loaded Nachos",
        "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQcdgvUozwgBB17AdzqgfIMZkKzCmHYM2uo1Q&s",
        "time": "15 minutes",
        "ingredients": [
            { "name": "tortilla chips", "amount": "2 cups" },
            { "name": "seasoned meat", "amount": "1/2 cup" },
            { "name": "cheese", "amount": "1/2 cup, shredded" },
            { "name": "tomatoes or salsa", "amount": "1/4 cup" },
            { "name": "onions", "amount": "2 tbsp, chopped" },
            { "name": "lime", "amount": "wedges for garnish" }
        ],
        "steps": [
            "Spread chips on a baking tray.",
            "Top with meat and cheese, then bake at 375°F for 10 minutes.",
            "Add salsa, onions, and a squeeze of lime before serving."
        ]
        }
    ]
    },


# 7) live_demo.jpeg (the new one with lemon, carrot, broccoli)
"live_demo.jpeg": {
  "ingredients": ["Lemon", "Carrot", "Broccoli"],
  "recipes": [
    {
      "title": "Roasted Broccoli & Carrots with Lemon Zest",
      "image_url": "https://www.onionringsandthings.com/wp-content/uploads/2017/10/lemon-garlic-roasted-broccoli-carrots-7.jpg",
      "time": "20 minutes",
      "ingredients": [
        { "name": "broccoli", "amount": "1 head" },
        { "name": "carrots", "amount": "1 large" },
        { "name": "lemon", "amount": "zest of 1" }
      ],
      "steps": [
        "Preheat oven to 400°F (200°C).",
        "Cut broccoli into florets and slice carrots.",
        "Spread on a baking sheet and drizzle with oil.",
        "Roast for 15–20 minutes until tender and slightly crispy.",
        "Sprinkle lemon zest over veggies before serving."
      ]
    },
    {
      "title": "Carrot-Broccoli Lemon Stir-Fry",
      "image_url": "https://i.ytimg.com/vi/kgbmKZ1GOxE/maxresdefault.jpg",
      "time": "15 minutes",
      "ingredients": [
        { "name": "carrot", "amount": "1, sliced" },
        { "name": "broccoli", "amount": "1 head, cut into florets" },
        { "name": "lemon", "amount": "juice of 1" }
      ],
      "steps": [
        "Heat oil in a pan over medium-high heat.",
        "Add sliced carrots and broccoli florets.",
        "Stir-fry for 5–7 minutes until tender-crisp.",
        "Add lemon juice, toss well, and cook for 1 more minute.",
        "Serve hot as a side or over rice."
      ]
    },
    {
      "title": "Broccoli-Carrot Soup with Lemon Twist",
      "image_url": "https://thefastrecipe.com/wp-content/uploads/2023/09/carrot-broccoli-soup-close-up.jpg",
      "time": "25 minutes",
      "ingredients": [
        { "name": "broccoli florets", "amount": "2 cups" },
        { "name": "carrots", "amount": "1–2, chopped" },
        { "name": "lemon", "amount": "juice (optional)" }
      ],
      "steps": [
        "In a pot, sauté chopped carrots and broccoli in a little oil for 3–4 minutes.",
        "Add enough water or broth to cover veggies and bring to a boil.",
        "Simmer for 15 minutes until vegetables are soft.",
        "Blend until smooth, add lemon juice (optional), and season to taste.",
        "Serve warm."
      ]
    },
    {
      "title": "Carrot & Broccoli Salad with Lemon Dressing",
      "image_url": "https://www.theendlessmeal.com/wp-content/uploads/2022/12/roasted-broccoli-and-carrots-4-1-750x750.jpg",
      "time": "10 minutes",
      "ingredients": [
        { "name": "broccoli florets", "amount": "1 cup, chopped" },
        { "name": "carrots", "amount": "1 shredded" },
        { "name": "lemon", "amount": "juice + zest" }
      ],
      "steps": [
        "In a bowl, combine chopped broccoli and shredded carrot.",
        "In a small bowl, mix lemon juice and zest with a pinch of salt.",
        "Toss veggies with the lemon dressing.",
        "Chill for 5 minutes or serve immediately."
      ]
    },
    {
      "title": "Steamed Veggies with Lemon Butter",
      "image_url": "https://cdn.apartmenttherapy.info/image/upload/v1642092369/k/Photo/Recipe%20Ramp%20Up/2022-01-Lemon-Butter-Sauce/kitchnoct_DSC4426-1.jpg",
      "time": "15 minutes",
      "ingredients": [
        { "name": "broccoli", "amount": "1 head, trimmed" },
        { "name": "carrot", "amount": "1 large, sliced" },
        { "name": "lemon", "amount": "half, juiced" }
      ],
      "steps": [
        "Steam broccoli and carrot slices for 5–7 minutes until tender.",
        "Melt butter in a small pan and stir in lemon juice.",
        "Drizzle lemon butter over steamed veggies.",
        "Serve warm."
      ]
    }
  ]
}

}
import pandas as pd
import json
import ollama
from pydantic import BaseModel, ValidationError
from typing import Literal
import re
from tqdm import tqdm
import os

'''
Steps :
X load des outfits
X augmentation des tags sur les outfits
- extraction des habits (pas besoin utiliser les extractions de base)
- augmentation des tags sur les habits (ça va être très simple)
- création d'une base d'outfit
- Embedding des habits
'''

OUTFITS_DATA_AUGMENTATION_PROMPT = 'Evaluate this image, your message NEEDING to follow the following JSON format :\n\
{\n\
"caption": [YOUR CAPTION], \n\
"type": \'Casual\'//\'Formal\'//\'Sports\'//\'Party\'/\'Home\', \n\
"weatherSuitability": \'Cold\'//\'Neutral\'//\'Warm\', \n\
}'

CLOTHES_DATA_AUGMENTATION_PROMPT = '''Evaluate this image. Your message MUST follow the JSON format below:

{
  "caption": [YOUR CAPTION AS STRING],
  "baseColour": "[One of: 'Navy Blue', 'Blue', 'Silver', 'Black', 'Grey', 'Green', 'Purple', 'White', 'Beige', 'Brown', 'Bronze', 'Teal', 'Copper', 'Pink', 'Off White', 'Maroon', 'Red', 'Khaki', 'Orange', 'Coffee Brown', 'Yellow', 'Charcoal', 'Gold', 'Steel', 'Tan', 'Multi', 'Magenta', 'Lavender', 'Sea Green', 'Cream', 'Peach', 'Olive', 'Skin', 'Burgundy', 'Grey Melange', 'Rust', 'Rose', 'Lime Green', 'Mauve', 'Turquoise Blue', 'Metallic', 'Mustard', 'Taupe', 'Nude', 'Mushroom Brown', 'Fluorescent Green']",
  "category": "[One of: 'Topwear', 'Bottomwear', 'Watches', 'Socks', 'Shoes', 'Belts', 'Flip Flops', 'Bags', 'Innerwear', 'Sandal', 'Shoe Accessories', 'Fragrance', 'Jewellery', 'Lips', 'Saree', 'Eyewear', 'Nails', 'Scarves', 'Dress', 'Loungewear and Nightwear', 'Wallets', 'Apparel Set', 'Headwear', 'Mufflers', 'Skin Care', 'Makeup', 'Free Gifts', 'Ties', 'Accessories', 'Skin', 'Beauty Accessories', 'Water Bottle', 'Eyes', 'Bath and Body', 'Gloves', 'Sports Accessories', 'Cufflinks', 'Sports Equipment', 'Stoles', 'Hair', 'Perfumes', 'Home Furnishing', 'Umbrellas', 'Wristbands', 'Vouchers']",
  "usage": "[One of: 'Casual', 'Formal', 'Sports', 'Smart Casual', 'Travel', 'Party', 'Home']"
}'''


DATA_AUGMENTATION_MODEL = 'llava'
DATASET_OUTFITS_PATH = 'data/clothing-coparsing-dataset/metadata.csv'
DATASET_OUTFITS_CLASS_DICT_PATH = 'data/clothing-coparsing-dataset/class_dict.csv'

RETRY_LIMIT = 1

def extract_json(text):

    pattern = r'\{.*?\}'

    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(0)
    
    return None

class OutfitCaption(BaseModel):
    caption: str
    type: Literal['Casual', 'Formal', 'Sports', 'Party', 'Home']
    weatherSuitability: Literal['Cold', 'Neutral', 'Warm']

class ClothesCaption(BaseModel):
    caption: str
    baseColour: Literal['Navy Blue', 'Blue', 'Silver', 'Black', 'Grey', 'Green', 'Purple',
       'White', 'Beige', 'Brown', 'Bronze', 'Teal', 'Copper', 'Pink',
       'Off White', 'Maroon', 'Red', 'Khaki', 'Orange', 'Coffee Brown',
       'Yellow', 'Charcoal', 'Gold', 'Steel', 'Tan', 'Multi', 'Magenta',
       'Lavender', 'Sea Green', 'Cream', 'Peach', 'Olive', 'Skin',
       'Burgundy', 'Grey Melange', 'Rust', 'Rose', 'Lime Green', 'Mauve',
       'Turquoise Blue', 'Metallic', 'Mustard', 'Taupe', 'Nude',
       'Mushroom Brown', 'Fluorescent Green']
    category: Literal['Topwear', 'Bottomwear', 'Watches', 'Socks', 'Shoes', 'Belts',
       'Flip Flops', 'Bags', 'Innerwear', 'Sandal', 'Shoe Accessories',
       'Fragrance', 'Jewellery', 'Lips', 'Saree', 'Eyewear', 'Nails',
       'Scarves', 'Dress', 'Loungewear and Nightwear', 'Wallets',
       'Apparel Set', 'Headwear', 'Mufflers', 'Skin Care', 'Makeup',
       'Free Gifts', 'Ties', 'Accessories', 'Skin', 'Beauty Accessories',
       'Water Bottle', 'Eyes', 'Bath and Body', 'Gloves',
       'Sports Accessories', 'Cufflinks', 'Sports Equipment', 'Stoles',
       'Hair', 'Perfumes', 'Home Furnishing', 'Umbrellas', 'Wristbands',
       'Vouchers']
    usage: Literal['Casual', 'Formal', 'Sports', 'Smart Casual',
       'Travel', 'Party', 'Home']

def call_model(model: str, system_prompt: str, prompt: str, image_path: str):
    response = ollama.chat(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': system_prompt            
            },
            {
                'role': 'user', 
                'content': prompt,
                'images': [image_path]
            },
        ],
    )
    return response.message.content


def load_outfits():
    df_metadata = pd.read_csv(DATASET_OUTFITS_PATH)
    return df_metadata[df_metadata["label_type"] == 'image-level']

def load_class_dict():
    return pd.read_csv(DATASET_OUTFITS_CLASS_DICT_PATH)

def outfits_data_augmentation(image_path: str):

    for i in range(RETRY_LIMIT):

        response = call_model(DATA_AUGMENTATION_MODEL, OUTFITS_DATA_AUGMENTATION_PROMPT, "", image_path)

        try:
            return OutfitCaption.model_validate_json(extract_json(response))
        except ValidationError as e:
            print(f"Validation ({i+1}/{RETRY_LIMIT} failed. Retrying...)")
            last_error = e
            pass

    raise last_error

def clothes_data_augmentation(image_path: str, prompt: str):

    for i in range(RETRY_LIMIT):

        response = call_model(DATA_AUGMENTATION_MODEL, CLOTHES_DATA_AUGMENTATION_PROMPT, prompt, image_path)

        try:
            return ClothesCaption.model_validate_json(extract_json(response))
        except ValidationError as e:
            print(f"Validation ({i+1}/{RETRY_LIMIT} failed. Retrying...)")
            last_error = e
            pass

    raise last_error

def data_integration_pipeline():

    outfits = load_outfits()
    outfits_class_dict = load_class_dict()

    # ----- dataset outfit augmentation -----
    if not os.path.isfile("checkpoints/outfits_first_augmentation.pkl"): 
        idx_to_drop = []

        for idx, row in tqdm(outfits.iterrows()):
            try:
                outfit_caption = outfits_data_augmentation(f'data/clothing-coparsing-dataset/{row["image_path"]}')
                outfits.at[idx, 'caption'] = outfit_caption.caption
                outfits.at[idx, 'type'] = outfit_caption.type

                print(f"Succesfully augmented row {idx}")

            except ValidationError:
                print(f"Validation failed after {RETRY_LIMIT} tries. Skipping row {idx}.")
                idx_to_drop.append(idx)
                continue

        outfits = outfits.drop(idx_to_drop)
        outfits.to_pickle("checkpoints/outfits_first_augmentation.pkl")
    else:
        outfits = pd.read_pickle("checkpoints/outfits_first_augmentation.pkl")
        

    # ----- converting outfit clothes to the clothes dataset -----
    df_clothes_from_outfits = pd.DataFrame(columns=['class_name', 'outfit_id', 'image_path'])

    outfits['clothes'] = None

    for index, row in outfits.iterrows():
        lab_path = f"data/clothing-coparsing-dataset/labels/{row['label_path']}"
        lab_file = open(lab_path, 'r')
        clothes_names = [outfits_class_dict["class_name"][int(line)] for line in lab_file.readlines() if int(line) != 0]

        clothes_id = []
        for cn in clothes_names:
            id_clothes = len(df_clothes_from_outfits)
            clothes_id.append(id_clothes)
            df_clothes_from_outfits.loc[id_clothes] = [cn, index, row['image_path']]
        
        outfits.at[index, 'clothes'] = clothes_id

    # ----- extract clothes image from outfit -----
    # here with a deeplearning model trained to identify clothes 
    # we could extract and crop the clothes into new images that could be processed 
    # more easily by the multimodal data augmentation LLM

    # ----- dataset outfit augmentation -----
    if not os.path.isfile("checkpoints/clothes_second_augmentation.pkl"):
        df_clothes_from_outfits['caption'] = None
        df_clothes_from_outfits['baseColour'] = None
        df_clothes_from_outfits['category'] = None
        df_clothes_from_outfits['usage'] = None

        for idx, row in tqdm(df_clothes_from_outfits.iterrows()):
            try:
                clothes_caption = clothes_data_augmentation(
                    f'data/clothing-coparsing-dataset/{row["image_path"]}', 
                    f'Focus on the tags for this piece of clothing : {row.class_name}'
                )
                df_clothes_from_outfits.at[idx, 'caption'] = clothes_caption.caption
                df_clothes_from_outfits.at[idx, 'baseColour'] = clothes_caption.baseColour
                df_clothes_from_outfits.at[idx, 'category'] = clothes_caption.category
                df_clothes_from_outfits.at[idx, 'usage'] = clothes_caption.usage

                print(f"Succesfully augmented row {idx}")

            except ValidationError:
                print(f"Validation failed after {RETRY_LIMIT} tries. Skipping row {idx}.")
                continue

        df_clothes_from_outfits.to_pickle("checkpoints/clothes_second_augmentation.pkl")
    else:
        df_clothes_from_outfits = pd.read_pickle("checkpoints/clothes_second_augmentation.pkl")

    print(outfits)
    print(df_clothes_from_outfits)
    return

data_integration_pipeline()
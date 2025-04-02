import pandas as pd
import json
import ollama
from pydantic import BaseModel, ValidationError
from typing import Literal
import re
from tqdm import tqdm

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

CLOTHES_DATA_AUGMENTATION_PROMPT = 'Evaluate this image, your message NEEDING to follow the following JSON format :\n\
{\n\
"caption": [YOUR CAPTION], \n\
"color": \'blue\'//\'cyan\'//\green\'//\black\'//\yellow\'//\red\'//\magenta\'//\white\', \n\
}'


DATA_AUGMENTATION_MODEL = 'llava'
DATASET_OUTFITS_PATH = 'data/clothing-coparsing-dataset/metadata.csv'
DATASET_OUTFITS_CLASS_DICT_PATH = 'data/clothing-coparsing-dataset/class_dict.csv'

RETRY_LIMIT = 3

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
    color: Literal['blue', 'cyan', 'green', 'black', 'yellow', 'red', 'magenta', 'white']    

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
    return df_metadata[df_metadata["label_type"] == 'image-level'].head(3)

def load_class_dict():
    return pd.read_csv(DATASET_OUTFITS_CLASS_DICT_PATH)

def outfits_data_augmentation(image_path: str):

    for i in range(RETRY_LIMIT):

        response = call_model(DATA_AUGMENTATION_MODEL, OUTFITS_DATA_AUGMENTATION_PROMPT, "", image_path)

        try:
            return OutfitCaption.model_validate_json(extract_json(response))
        except ValidationError:
            print(f"Validation ({i+1}/{RETRY_LIMIT} failed. Retrying...)")
            pass

    raise ValidationError

def clothes_data_augmentation(image_path: str):

    for i in range(RETRY_LIMIT):

        response = call_model(DATA_AUGMENTATION_MODEL, CLOTHES_DATA_AUGMENTATION_PROMPT, "", image_path)

        try:
            return ClothesCaption.model_validate_json(extract_json(response))
        except ValidationError:
            print(f"Validation ({i+1}/{RETRY_LIMIT} failed. Retrying...)")
            pass

    raise ValidationError

def data_integration_pipeline():

    outfits = load_outfits()
    outfits_class_dict = load_class_dict()  # Make an augmented row too?

    idx_to_drop = []

    # dataset outfit augmentation
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

    # df_clothes_from_outfits = []

    # for index, row in outfits.iterrows():
    #     pass
        
    # oskour
    # for h in outfits.habits:
    #     outfits['clothes'] = [id des clothes]
    #     df_clothes_from_outfits.append("", "", "", "")


    # clothes_data_augmentation()
    return

data_integration_pipeline()
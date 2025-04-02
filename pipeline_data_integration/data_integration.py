import pandas as pd
import ollama
from pydantic import BaseModel, ValidationError
from typing import Literal

'''
Steps :
- load des outfits
- augmentation des tags sur les outfits
X extraction des habits (pas besoin utiliser les extractions de base)
- augmentation des tags sur les habits
- création d'une base d'outfit
'''

OUTFITS_DATA_AUGMENTATION_PROMPT = ''
CLOTHES_DATA_AUGMENTATION_PROMPT = ''
DATA_AUGMENTATION_MODEL = 'llava'
DATASET_OUTFITS_PATH = 'data/clothing-coparsing-dataset/metadata.csv'
DATASET_OUTFITS_CLASS_DICT_PATH = 'data/clothing-coparsing-dataset/class_dict.csv'

class OutfitCaption(BaseModel):
    caption: str
    type: Literal['Casual', 'Formal', 'Sports', 'Party', 'Home']

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
    return df_metadata[df_metadata["label_type"] == 'image-level']

def load_class_dict():
    return pd.read_csv(DATASET_OUTFITS_CLASS_DICT_PATH)

def outfits_data_augmentation(image_path: str):
    response = call_model(DATA_AUGMENTATION_MODEL, OUTFITS_DATA_AUGMENTATION_PROMPT, "", image_path)

    try:
        return response.model_validate_json()
    except ValidationError:
        print('oskour')

def clothes_data_augmentation(image_path: str):
    response = call_model(DATA_AUGMENTATION_MODEL, CLOTHES_DATA_AUGMENTATION_PROMPT, "", image_path)

    try:
        return response.model_validate_json()
    except ValidationError:
        print('oskour2')

def data_integration_pipeline():
    outfits = load_outfits()
    outfits_class_dict = load_class_dict()

    # dataset outfit augmentation
    for index, row in outfits.iterrows():   
        outfit_caption = outfits_data_augmentation(f"data/clothing-coparsing-dataset/{row["image_path"]}")
        outfits['caption'] = outfit_caption.caption
        outfits['type'] = outfit_caption.type

    df_clothes_from_outfits = []

    for index, row in outfits.iterrows():
        
    # oskour
    # for h in outfits.habits:
    #     outfits['clothes'] = [id des clothes]
    #     df_clothes_from_outfits.append("", "", "", "")


    clothes_data_augmentation()
    return

import os
import io
from openai import OpenAI
import base64


IMAGE_DIR = "/home/semanticnuc/Desktop/Tiago/TIAGo-RoCoCo/KG_Reasoning/vlm_grasping/images/scans/"
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

prompt = "You are an assistent which is able accurately describe the content of an image. \
                    In particular, you are able to capture the main objects present \
                    in the image and provide the relations that exist between them. \
                    These relations are described in the form of a triple (subject, relation, object) \
                    and when you answer you are only expected to answer with triples and nothin else. \
                    When considering positional relations like 'right to' or 'left to', \
                    assume that the camera is your point of view.   \
                    For example, if in a scene there is a door, a table in front of the door and a book on the table \
                    with a pen right to it, your answere should be: \
                    1) (table, in front of, door) \
                    2) (book, on, table) \
                    3) (book, on, table) \
                    4) (pen, right to, book) "

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_captions(image_path):

    image = encode_image(image_path)
    response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
    {
        "role": "user",
        "content": [
        {
            "type": "text",
            "text": f"{prompt}",
        },
        {
            "type": "image_url",
            "image_url": {
            "url": f"data:image/jpeg:base64.{image}",
            },
        },
        ],
    }
    ],
    max_tokens=300,
    temperature=0,
    )

    return response.choices[0].message.content

images = os.listdir(IMAGE_DIR)
images.sort()

for image_ in images:
#    print(IMAGE_DIR+image_)
    captions = get_captions(IMAGE_DIR+image_)
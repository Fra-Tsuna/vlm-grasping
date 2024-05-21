from openai import OpenAI
client = OpenAI()


class Agents:
    def __init__(self, image, task_description):
        self.encoded_image = image
        self.task_description = task_description
    
    def vision_agent(self):
            agent = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": f"You are an agent which is very capable to describe the objects within an environtment. \n \
                            You are give in input an image that depicts the scene and a task description of what you are going to do next. \n \
                            From the image, you need to produce as output as set of relations in the form of tirple (subject, relation, object) \
                            of all the objects that you think they are meaningful in order to solve the task. \n \
                            The task is: {self.task_description}. \n \
                            Use specific relations to describe the position of the objects in the scene. Don't use arbitrary words like 'next', but rather \
                            'right to' or 'on', considering the camera point of view. \n \
                            For example, if in a scene there is a door, a table in front of the door and a book on the table \
                            with a pen right to it, your answer should be: \n \
                            1) (table, in front of, door) \n\
                            2) (book, on, table) \n\
                            3) (pen, on, table) \n\
                            4) (pen, right to, book). \n\
                            "
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encoded_image}",
                    },
                    },
                ],
                }
            ],
            max_tokens=500,
            temperature=0,
            )
            return agent.choices[0].message.content

import base64
from agents import Agents

# task_description = "order the objects on the table so that they are from highest to lowest on the same line"
#task_description = "unplug the electric plug from the socket"
# task_description = "exit the room moving the object in the trajectory if it is necessary"
# task_description = "throw away the objects on the floor in the corresponding recycling bin"
# task_description = "put the cups one inside the other by using the one in the front as the base"
# task_description = "order the shelf to have just two objects for each level of the shelf"
task_description = "give me the green jacket from the clothing rack"

image_path = "/home/semanticnuc/Desktop/Tiago/TIAGo-RoCoCo/KG_Reasoning/vlm-grasping/images/test_order/rgb.jpg"
print(task_description[0:10])
with open(image_path, "rb") as im_file:
    encoded_image = base64.b64encode(im_file.read()).decode("utf-8")
        
agents = Agents(encoded_image,task_description)

import pickle
single_agen = False
if single_agen:

    single_agent_info = agents.single_agent() 
    print(single_agent_info)
    results_single = {
        "single_agent_info": single_agent_info
    }
    with open('results_single_' + task_description[0:10] +"_" +image_path.replace(".jpg","") + '.pkl', 'wb') as f:
        pickle.dump(results_single, f)

else:
    enviroment_info, description_agent_info, planning_agent_info = agents.multi_agent_vision_planning()

    results_multi = {

        "enviroment_info": enviroment_info,
        "description_agent_info": description_agent_info,
        "planning_agent_info": planning_agent_info,
    }
    with open('/home/semanticnuc/Desktop/Tiago/TIAGo-RoCoCo/KG_Reasoning/vlm-grasping/config/agents/results_dif_final.pkl', 'wb') as f:
        pickle.dump(results_multi, f,protocol=2)

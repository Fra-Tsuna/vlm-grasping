import pickle


with open('results_ord_final.pkl', 'rb') as file:
    data = pickle.load(file)
 
print(data['enviroment_info'])  

print("\n______________________\n")

print(data['description_agent_info'])

print("\n______________________\n")

print(data['planning_agent_info'])

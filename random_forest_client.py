import requests
import json

def request_prediction(instance):
    list_q = ["treatment","tech_company","wellness_program","seek_help","anonymity","leave","phys_health_consequence", "coworkers","supervisor", "mental_health_interview", "mental_vs_physical", "obs_consequence"]
    list_aq = []
    url = "http://localhost:5000/predict?"
    for i,q in enumerate(list_q):
        if i > 0:
            list_aq.append("&" + q + "=" + instance[i])
        else:
            list_aq.append(q + "=" + instance[i])
    for inst in list_aq:
        url += inst
    print(url)

    # open the URL and read the server's response
    response = requests.get(url=url)
    json_object = json.loads(response.text)
    print("response:", json_object)
    return json_object["prediction"]

unseen_instance = ['Yes', 'Yes', 'No', 'Yes', 'Yes', 'Somewhat easy', 'No', 'Some of them', 'Yes', 'No', 'Yes', 'No']
#for unseen_instance in X_test:
prediction = request_prediction(unseen_instance)
print("prediction for", unseen_instance, ":", prediction)
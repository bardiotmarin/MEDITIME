import requests

def find_specialist(bot_response):
    # Extract important information from bot response (e.g., health concern)
    # You may need to add NLP parsing here for better accuracy.
    health_concern = extract_health_concern(bot_response)

    # requette api 
    url = f"https://public.opendatasoft.com/api/records/1.0/search/?dataset=medecins&q={health_concern}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data["nhits"] > 0:
            # retourne le spé 
            return data["records"][0]["fields"]["nom_du_medecin"]
        else:
            return "No specialists found for the given concern."
    else:
        return "API request failed."

def extract_health_concern(text):
    # Basic text parsing, this can be improved with NLP models
    # For now, let's just assume the last word is the health issue
    return text.split()[-1]

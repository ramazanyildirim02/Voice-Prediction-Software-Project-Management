import joblib

def predict(value):
    model = joblib.load("model.sav")
    prediction =  model.predict(value)

    
    if prediction[0] == "ceylin": 
        return "Ses Ceylin'e Ait"
    
    elif prediction[0] == "semih":
        return "Ses Semih'e Ait"
    
    elif prediction[0] == "ramazan":
        return "Ses Ramazan'a Ait"
    
    elif prediction[0] == "hasan":
        return "Ses Hasan'a Ait"
    
    else:
        return "Bu Ses Tanıdık Değil!"

  
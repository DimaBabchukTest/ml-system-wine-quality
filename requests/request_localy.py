# uv run python requests/request_localy.py
import requests

client1 = {
   "fixed_acidity":"7.4",
   "volatile_acidity":"0.7",
   "citric_acid":"0.0",
   "residual_sugar":"1.9",
   "chlorides":"0.076",
   "free_sulfur_dioxide":"11.0",
   "total_sulfur_dioxide":"34.0",
   "density":"0.9978",
   "ph":"3.51",
   "sulphates":"0.56",
   "alcohol":"9.4",
   "quality":"5"
}

client2 = {
      "fixed_acidity":7.8,
      "volatile_acidity":0.76,
      "citric_acid":0.04,
      "residual_sugar":2.3,
      "chlorides":0.092,
      "free_sulfur_dioxide":15.0,
      "total_sulfur_dioxide":54.0,
      "density":0.997,
      "ph":3.26,
      "sulphates":0.65,
      "alcohol":9.8,
      "quality":5
   }
client3 = {
      "fixed_acidity":11.2,
      "volatile_acidity":0.28,
      "citric_acid":0.56,
      "residual_sugar":1.9,
      "chlorides":0.075,
      "free_sulfur_dioxide":17.0,
      "total_sulfur_dioxide":60.0,
      "density":0.998,
      "ph":3.16,
      "sulphates":0.58,
      "alcohol":9.8,
      "quality":6
   }
client4 = {
      "fixed_acidity":6.5,
      "volatile_acidity":0.24,
      "citric_acid":0.19,
      "residual_sugar":1.2,
      "chlorides":0.041,
      "free_sulfur_dioxide":30.0,
      "total_sulfur_dioxide":111.0,
      "density":0.99254,
      "ph":2.99,
      "sulphates":0.46,
      "alcohol":9.4,
      "quality":6
   }

client5 = {
      "fixed_acidity":5.5,
      "volatile_acidity":0.29,
      "citric_acid":0.3,
      "residual_sugar":1.1,
      "chlorides":0.022,
      "free_sulfur_dioxide":20.0,
      "total_sulfur_dioxide":110.0,
      "density":0.98869,
      "ph":3.34,
      "sulphates":0.38,
      "alcohol":12.8,
      "quality":7
   }

url = "http://127.0.0.1:8000/predict"
print ('Client 1 -', requests.post(url, json=client1).json())
print ('Client 2 -', requests.post(url, json=client2).json())
print ('Client 3 -', requests.post(url, json=client3).json())
print ('Client 4 -', requests.post(url, json=client3).json())
print ('Client 5 -', requests.post(url, json=client3).json())

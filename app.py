import flasgger
from flasgger import Swagger
from flask import Flask, request
import joblib
import pandas as pd
import numpy as np

#Initiate app
app = Flask(__name__)
Swagger(app)



#Load regression model
model = joblib.load('classifier.joblib')



#Launching welcome page
@app.route('/')
def Welcome():
   return "welcome all"

#craeting a swagger API
 
#creating prediction for the values in csv file
@app.route('/predict_file',methods = ["POST"])
def predictFile():

    '''API to predict customer churn
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
    responses:
          200:
              description: The output values
    '''


    df = pd.read_csv(request.files.get('file'))
    prediction = model.predict(df)
    return 'Your predicted salary values for csv are:'+ str(list(prediction))



    
    
if __name__ == '__main__':
   app.run(debug=True, port=5001)

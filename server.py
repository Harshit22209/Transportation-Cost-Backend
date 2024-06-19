from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from datetime import datetime
import tensorflow as tf
import numpy as np
import joblib
from pulp import *
app = Flask(__name__)
scaler = joblib.load('model/scaler.pkl')
target_scaler=joblib.load('model/target_scaler.pkl')
CORS(app)  # This will enable CORS for all routes
from tensorflow.keras.models import load_model


# Load the model with custom objects specified
model = load_model('model/lstm_model.h5')
def predict_order_demand(input_data):
  input_10=[[input_data[0],input_data[1],input_data[2],input_data[3]] for i in range(10)]
  new_data_scaled = scaler.transform(input_10)

  # Create a sequence from the new data
  time_step = 10  # Use the same time step used during training
  new_X = new_data_scaled[-time_step:]  # Take the last `time_step` rows
  new_X = new_X.reshape(1, time_step, new_X.shape[1])  # Reshape for LSTM

  # Load the model

  # Make a prediction
  predicted_value_scaled = model.predict(new_X)

  # Inverse transform the prediction to the original scale
  predicted_value = target_scaler.inverse_transform(predicted_value_scaled)

  print(predicted_value)
  return predicted_value[0][0]
def minimize_transport_cost(day,month,year):
    Warehouses = ["A", "B"]

    # Creates a dictionary for the number of units of supply for each supply node
    supply = {"A": 20000, "B": 18000}

    # Creates a list of all demand nodes
    Bars = ["1", "2", "3", "4"]

    # Creates a dictionary for the number of units of demand for each demand node
    demand = {
        "1": float(predict_order_demand([1, day, month, year])),
        "2": float(predict_order_demand([2, day, month, year])), 
        "3": float(predict_order_demand([3, day, month, year])),
        "4": float(predict_order_demand([4, day, month, year])),
    }
    
    # Creates a list of costs of each transportation path
    costs = [  # Bars
        # 1 2 3 4 5
        [2, 4, 5, 2],  # A   Warehouses
        [3, 1, 3, 2],  # B
    ]

    # The cost data is made into a dictionary
    costs = makeDict([Warehouses, Bars], costs, 0)

    # Creates the 'prob' variable to contain the problem data
    prob = LpProblem("Transport Cost Prob", LpMinimize)

    # Creates a list of tuples containing all the possible routes for transport
    Routes = [(w, b) for w in Warehouses for b in Bars]

    # A dictionary called 'Vars' is created to contain the referenced variables(the routes)
    vars = LpVariable.dicts("Route", (Warehouses, Bars), 0, None, LpInteger)

    # The objective function is added to 'prob' first
    prob += (
        lpSum([vars[w][b] * costs[w][b] for (w, b) in Routes]),
        "Sum_of_Transporting_Costs",
    )

    # The supply maximum constraints are added to prob for each supply node (warehouse)
    for w in Warehouses:
        prob += (
            lpSum([vars[w][b] for b in Bars]) <= supply[w],
            f"Sum_of_Products_out_of_Warehouse_{w}",
        )

    # The demand minimum constraints are added to prob for each demand node (bar)
    for b in Bars:
        prob += (
            lpSum([vars[w][b] for w in Warehouses]) >= demand[b],
            f"Sum_of_Products_into_Retail{b}",
        )

    # The problem data is written to an .lp file
    prob.writeLP("TransportCostProb.lp")

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    # Each of the variables is printed with it's resolved optimum value
    out=[]
    for v in prob.variables():
        print(v.name, "=", v.varValue)
        out.append(int(v.varValue))

    # The optimised objective function value is printed to the screen
    print("Total Cost of Transportation = ", value(prob.objective))
    return (demand,out,int(value(prob.objective)))

@app.route('/submit-date', methods=['POST'])
def submit_date():
    data = request.json
    date = data.get('date')
    # Process the date as needed (e.g., save to database)
    date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Extract day of the week, month, and year
    weekday = date_obj.isoweekday()  # Monday is 1, Sunday is 7
    month = date_obj.month
    year = date_obj.year
    
    return jsonify({"result":minimize_transport_cost(weekday,month,year) , "date": date}), 200


if __name__ == '__main__':
    app.run(debug=True)

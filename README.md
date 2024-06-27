# Transport Cost Optimization

## Introduction

This project involves the development of an optimization tool and machine learning model to minimize transportation costs. By predicting the demand for each retail shop and solving a linear programming transportation problem, the solution aims to determine the optimal distribution of products from warehouses to retailers.

![image](https://github.com/Harshit22209/Transportation-Cost-Backend/assets/119040511/d1579201-80fc-4b55-b40c-5ec77c8d125f)


## Features

- **Demand Prediction**: Utilizes an LSTM model to forecast the demand for each retail shop.
- **Linear Programming**: Models a transportation problem with constraints on supply and demand.
- **Cost Optimization**: Implements a solution in Python to find the optimal distribution of products.

## Technologies Used

- **Programming Language**: Python
- **Machine Learning**: LSTM Model
- **Optimization**: Linear Programming
- **Frontend**: React

## Project Details

### Demand Prediction

- Developed an LSTM model to accurately predict the demand for each retail shop.
- The model takes historical sales data as input and forecasts future demand.

### Linear Programming

- With the predicted demand, a linear programming model is created to solve the transportation problem.
- Constraints include:
  - Supply constraints at warehouses.
  - Demand constraints at retail shops.

### Cost Optimization

- The optimization tool is implemented in Python.
- The tool determines the optimal distribution of products from warehouses to retailers, minimizing transportation costs.


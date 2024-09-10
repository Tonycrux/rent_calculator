# Rent Prediction Flask App

This project is a Flask web application that predicts apartment rent prices based on input features such as the number of bedrooms, bathrooms, and location. The app uses a pre-trained machine learning model (Random Forest Regressor) to make predictions, with the backend written in Python and the model built using scikit-learn.

## Features

- Predict rent prices for apartments based on the following factors:
  - Number of Bedrooms
  - Number of Bathrooms
  - Number of Toilets
  - Apartment Status: Newly Built, Furnished, and/or Serviced
  - City
- Simple and intuitive web interface for users to input data.
- Scalable Flask backend that processes the data and returns predictions.

## Technologies Used

- **Flask**: Web framework for building the API.
- **Python**: Backend programming language.
- **scikit-learn**: Machine learning library used to train the model.
- **Pandas**: Data manipulation and analysis library.
- **NumPy**: Library for numerical operations.
- **Jinja**: Template engine for rendering HTML.
  
## Setup Instructions

### Prerequisites

Before you begin, ensure you have the following installed on your machine:

- [Python 3.x](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>

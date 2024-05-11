# Airbnb Booking Classifier

## Problem: 
this challenge, you are given a list of users along with their demographics, web session records, and some summary statistics. You are asked to predict which country a new user's first booking destination will be. All the users in this dataset are from the USA.
There are 12 possible outcomes of the destination country: 'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL','DE', 'AU', 'NDF' (no destination found), and 'other'. Please note that 'NDF' is different from 'other' because 'other' means there was a booking, but is to a country not included in the list, while 'NDF' means there wasn't a booking.

## Running
1. Install all the required Python packages using requirements.txt.
2. Move to the root directory of the project.
3. Execute the command ‘flask --app src/webserver/app:app run --host=0.0.0.0 --port=5555’.
4. Go on the browser and check ‘http://0.0.0.0:5555/’.


## Testing
1. To check single or multiple samples use http://127.0.0.1:5555/predict  endpoint, with the following format json.
{
    "data":{"distance_km": 0.0, "gender": "MALE", "signup_flow": 3.0, "account_created_year": 2010.0, "destination_language ": "eng", "affiliate_provider": "google", "first_device_type": "Windows Desktop", "timestamp_first_active": 20100804190250.0, "language": "en", "signup_method": "facebook", "first_affiliate_tracked": "linked", "account_created_month": "August", "population_in_thousands": 1743.1333333333334, "first_browser": "Firefox", "secs_elapsed": 872862.0, "first_booking_month": "June", "affiliate_channel": "seo", "age": 32.0, "first_booking_year": 2013.0},
    "n_data": 1
}
2. To run prediction on a file use http://127.0.0.1:5555/predict_batch. It will take a CSV file with the file as the key name.

## Future improvement
1. Try with multiple algorithms and different hyperparameters.
2. Model and feature vector registry for versioning and re-usability.
3. Saving and loading data from the cloud.
4. Deployment using Kubernetes to leverage autoscaling.
5. Monitor the model to check data shift.
6. Automated pipeline using AWS or other cloud ML service.





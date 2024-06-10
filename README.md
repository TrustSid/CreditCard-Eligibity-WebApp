# Credit Card Eligibility Application

A web-based solution to determine an individual's eligibility for a credit card from a bank. Built with a user-friendly front-end (HTML, CSS, JavaScript) and a robust back-end (Python, Flask). The application leverages machine learning models to analyze user data and provide a clear result on their credit card eligibility. Containerized using Docker for easy deployment and portability across environments.

## Features

- User-friendly form to input personal and financial information
- Backend logic to determine credit card eligibility based on the provided data
- Containerized deployment using Docker for easy setup and portability
  
## Getting Started

1. Clone the repository:
2. Navigate to the project directory:
3. Build the Docker images: Dockerfile (main directory),  Dockerfile (in templates directory)
## Build the front-end image
- docker build -t my-frontend-image .
## Build the back-end image
docker build -t my-backend-image .
- Note:** Build images only after selecting the right directories in CMD
## Run the front-end container
- docker run -d -p 8888:5000 my-frontend-image
## Run the back-end container
- docker run -d -p 5000:5000 my-backend-image

- Access the application by opening `http://localhost:8080` in your web browser.


  

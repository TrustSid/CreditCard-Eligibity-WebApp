# Credit Card Eligibility Application

This project is a web application that helps users check their eligibility for a credit card from a bank. It consists of a front-end interface built with HTML, CSS, and JavaScript, and a back-end component built using Python and Flask.

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
## Build the back-end image**
docker build -t my-backend-image .
- Note:** Build images only after selecting the right directories in CMD
## Run the front-end container
- docker run -d -p 8888:5000 my-frontend-image
## Run the back-end container
- docker run -d -p 5000:5000 my-backend-image

- Access the application by opening `http://localhost:8080` in your web browser.


  

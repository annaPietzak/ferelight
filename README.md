# FERElight | ˈferēlīt |
Extremely lightweight and purpose-built feature extraction and retrieval engine (FERE).

## Usage
To run the server, please execute the following from the root directory:

```
pip3 install -r requirements.txt
python3 -m ferelight
```

## Running with Docker

To run the server on a Docker container, please execute the following from the root directory:

```bash
# building the image
docker build -t ferelight .

# starting up a container
docker run -p 8080:8080 ferelight
```
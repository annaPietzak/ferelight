# FERElight | ˈferēlīt |
Extremely lightweight and purpose-built feature extraction and retrieval engine (FERE).

## Usage
To configure the pgvector PostgreSQL connection, create a file `config.json` in the root directory with the following content:

```json
{
  "DBHOST": "<host>",
  "DBPORT": "<port>",
  "DBUSER": "<user>",
  "DBPASSWORD": "<password>"
}
```

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
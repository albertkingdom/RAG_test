# RAG Demo Project

This project is a demonstration of a Retrieval-Augmented Generation (RAG) pipeline using LangChain, OpenAI, and Qdrant. The application fetches content from a web page, processes it, stores it in a Qdrant vector database, and answers questions based on the retrieved content.

## Features

- **Web Content Fetching**: Loads text data from a specified URL using `WebBaseLoader`.
- **Text Splitting**: Splits the document into smaller chunks for effective embedding and retrieval.
- **Vector Storage**: Uses Qdrant as the vector store for the document embeddings.
- **RAG Pipeline**: Implements a retrieval and generation chain using LangChain and OpenAI to answer questions.
- **Containerized Environment**: All services (application, Qdrant, Jupyter) are managed by Docker and Docker Compose for easy setup and execution.

## Requirements

- Docker
- Docker Compose

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd RAG
    ```

2.  **Create the environment file:**
    Create a `.env` file in the root of the project and add your API key:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    APMIC_API_KEY=APMIC自架模型APIKEY
    ```

3.  **Build and start the services:**
    Use Docker Compose to build the images and start all the containers in the background.
    ```bash
    docker-compose up -d --build
    ```

## Development with Dev Containers

This project is configured to use [Visual Studio Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) for a seamless and consistent development experience.

### Prerequisites

-   [Visual Studio Code](https://code.visualstudio.com/)
-   The [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code.
-   [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### Steps to Get Started

1.  **Open the project in VS Code:**
    After cloning the repository, open the project folder in Visual Studio Code.

2.  **Reopen in Container:**
    VS Code will detect the `.devcontainer/devcontainer.json` file and show a notification prompting you to "Reopen in Container". Click it.
    
    Alternatively, you can open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`) and select **"Dev Containers: Reopen in Container"**.

3.  **Start Developing:**
    VS Code will build the development container based on the `docker-compose.yml` configuration. Once it's finished, you'll be inside a fully configured environment. The source code is mounted into the container, so any changes you make are reflected on your local filesystem.
    
    You can now open a terminal directly in VS Code (`Ctrl+` or `Cmd+``) and run your application without needing to use `docker-compose exec`:
    ```bash
    python app.py
    ```

## Usage

### Running the RAG Application

To execute the main application script, run the following command:

```bash
docker-compose exec app python app.py
```

This will run the `app.py` script inside the `app` container, which performs the full RAG pipeline and prints the result to the console.

### Using Jupyter Notebook

A Jupyter Notebook environment is also available for interactive development and exploration.

-   **URL**: [http://localhost:8888](http://localhost:8888)
-   The environment has all the required Python packages installed.

## Project Structure

```
.
├── .env                # Environment variables (contains API key)
├── app.py              # Main application script for the RAG pipeline
├── docker-compose.yml  # Defines and configures all services
├── Dockerfile          # Dockerfile for the main application service
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Python Dependencies

The Python dependencies for this project are listed in `requirements.txt`:

- `langchain_openai`
- `langchain`
- `langchain_community`
- `bs4`
- `qdrant-client`

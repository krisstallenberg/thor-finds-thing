# AI2Thor finds things using chat

Group project for the **Communicative Robots** VU master's course taught from the [CLTL lab](http://www.cltl.nl).

## Installation

### Prerequisites

For this project, you need Ollama to locally run the LLama3.2 model:

1. [install Ollama](https://ollama.com/download) 
2. Install the **Llama3.2** model:
    ```bash
    ollama pull llama3.2
    ```

### Setting up

1. To create a virtual environment using Conda and install all dependencies, run:
    ```
    make
    ```
    > To avoid naming conflicts in your environment, we create not a global Conda environment, but a local one, in the `./myenv` directory.

2. Next, add your OpenAI API key to the `.env` file:
    ```
    OPENAI_API_KEY=sk-proj...
    ```

## Usage

The main application is a [Chainlit](https://docs.chainlit.io/get-started/overview) app.

To run this app from the virtual environment, run:

```
make run
```

### Development

To start an interactive Python notebook from the virtual environment, run:

```
make jupyter
```

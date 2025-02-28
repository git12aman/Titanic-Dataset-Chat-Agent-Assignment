# Titanic Dataset Chatbot

This project is a Titanic dataset chatbot that allows users to interact with the dataset using natural language queries and view visualizations. It is built using **FastAPI** for the backend and **Streamlit** for the frontend.

## Features

- **Chat with the Titanic Dataset** using LangChain and OpenAI's GPT API.
- **FastAPI Backend** to handle requests for textual analysis and visualizations.
- **Streamlit Frontend** for a user-friendly chat and visualization interface.
- **Dynamic Data Visualizations** using Matplotlib and Seaborn.

## Tech Stack

- **Backend**: FastAPI, LangChain, OpenAI API, Pandas
- **Frontend**: Streamlit
- **Visualization**: Matplotlib, Seaborn

## Installation

### Prerequisites

- Python 3.8+
- Pip
- Google gemini API Key
- 
### Install Dependencies

```sh
pip install -r requirements.txt
```

### Set Up Environment Variables

Create a `.env` file and add your OpenAI API key:

```sh
OPENAI_API_KEY=your_openai_api_key
```

## Running the Application

### Start the FastAPI Backend

```sh
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

### Start the Streamlit Frontend

```sh
streamlit run frontend.py
```

## API Endpoints

| Method | Endpoint             | Description                                                         |
| ------ | -------------------- | ------------------------------------------------------------------- |
| POST   | `/api/chat`          | Accepts a query and returns a response based on the Titanic dataset |
| GET    | `/api/visualization` | Returns a Titanic dataset visualization                             |

## Example Questions

- "How many passengers survived?"
- "What was the average age of passengers?"
- "Show me a survival rate chart by class."

## Screenshots



## Contributing

1. Fork the repo
2. Create a new branch (`feature-name`)
3. Commit changes
4. Push to branch
5. Create a pull request

## License

This project is licensed under the MIT License.

---


**GitHub:** [Your GitHub Profile](https://github.com/git12aman)


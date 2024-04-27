# Doc-visor
An LLM solution for querying based on your data.

Doc-visor is a trade query advisor solution that leverages natural language processing (NLP) and Large Language Models (LLMs) to provide insights and answers to trade-related queries.

## Overview

Tradevisor offers the following key features:

- **Document Indexing**: Indexes trade documents for efficient retrieval and analysis.
- **Milvus Integration**: Utilizes Milvus for vector storage and retrieval.
- **Query Handling**: Handles trade-related queries using advanced NLP models.
- **Chat Interface**: Provides a chat interface for interactive querying.
- **Persistent Logging**: Logs conversation history for reference and analysis.

## Installation

Follow these steps to install and set up Tradevisor:

1. Clone the repository:

   ```bash
   git clone https://github.com/NovaXLink/tradxlink-tradvisor.git
   ```

2. Navigate to the project directory:

   ```bash
   cd tradxlink-tradvisor
   ```

3. Install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use Tradvisor, follow these steps:

1. Run the Flask application:

   ```bash
   python app.py
   ```

2. After running the application, you can access the API endpoints using a tool like Postman or by making HTTP requests from your application.

## Endpoints

<b>1. Initialize Trade LLM: /trade_llm/init_trade_llm (POST)</b>

- Input - json request body

   ```json request body
    {
    "path": "Directory of the text files generated",
    "update": "Flag to indicate whether to update existing indexes"
    }
   ```

- Output - JSON Response

   - Success: `{'message': 'Trade LLM Initialization Successful'}`
   - Error: `{'error': <error_message>}` (HTTP status code 500)

<b>2. Convert PDF to Text: /trade_llm/pdf_to_txt (GET)</b>

- Input - json request body

   ```json request body
    {
    "directory_name": "Name of the directory to store text files",
    "pdf_directory": "Directory containing PDF files",
    "refresh_content": "Flag to indicate whether to refresh existing files"
    }
   ```

- Output - JSON Response

   - Success: `{'text file created successfully': True}`
   - Error: `{'error': <error_message>}` (HTTP status code 500)   

<b>3. Initialize Milvus: /trade_llm/init_milvus (GET)</b>

- Input - None

- Output - JSON Response

   - Success: `{'message': 'Initialization successful'}`
   - Error: `{'error': <error_message>}` (HTTP status code 500) 

<b>4. Query Trade Information: /trade_llm/query (POST)</b>

- Input - json request body

   ```json request body
    {
    "corelationId": "Correlation ID associated user",
    "query": "Query string",
    "country": "Country information",
    }
   ```

- Output - JSON Response

   - Success: `{'message': 'Success', 'query_result': <query_result>}`
   - Error: `{'error': <error_message>}` (HTTP status code 500)

<b>5. Reset Conversation Log: /trade_llm/reset_conversation (GET)</b>

- Input - json request body

   ```json request body
    {
    "corelationId": "Correlation ID for identifying the conversation log to reset"
    }
   ```

- Output - JSON Response

   - Success: `{'message': 'Conversation log reset successfully'}`
   - No matching corelationId found: `{'message': 'No matching corelationId found'}`
   - Error: `{'error': <error_message>}` (HTTP status code 500)

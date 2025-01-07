# FastAPI Data Mapping API

This API is designed for data mapping and processing tasks, allowing inference, caching, and management of input data. The API features include efficient caching, real-time inference, and cache management endpoints.

## Features

- **Data Mapping Endpoint**: Processes input data to generate mappings and unmapped indices.
- **Caching**: Utilizes a caching mechanism to improve performance by avoiding redundant computations.
- **Cache Management**: Provides an endpoint to flush all cached data.
- **Performance Metrics**: Measures and returns timing information for inference and API calls.

## Requirements

- Python 3.8+
- Required Python packages:
  - `fastapi`
  - `pydantic`
  - `torch`
  - `pandas`
  - `uvicorn`
  - huggingface and tokenizer libraries

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

## Endpoints

### 1. `/data_mapping/` [POST]
Processes input data, performs inference, and returns the mapping results.

#### Request Body
```json
[
    {
        "ships_idx": 1,
        "ship_data_list": [
            {
                "index": 1,
                "tag_name": "Tag1",
                "equip_type_code": 101,
                "tag_description": "Description1",
                "tx_period": 10,
                "tx_type": 1,
                "on_change_yn": true,
                "scaling_const": 1.0,
                "signal_type": "Type1",
                "min": 0.0,
                "max": 100.0,
                "unit": "Unit1",
                "data_type": 2
            }
        ]
    }
]
```

#### Response
```json
{
    "message": "Data mapped successfully",
    "result": [
        {
            "ships_idx": 1,
            "platform_data_list": [
                {"index": 1, "thing": "Thing1", "property": "Property1"}
            ],
            "unmapped_indices": []
        }
    ],
    "timing": {
        "total_inference_time": 0.12,
        "total_api_time": 0.2
    }
}
```

### 2. `/data_mapping_cached/` [POST]
Similar to `/data_mapping/`, but utilizes caching for improved performance.

### 3. `/flush_cache/` [DELETE]
Flushes all cached data.

#### Response
```json
{
    "message": "Flushed 5 cache files successfully."
}
```

## How to Run the API (Test)

1. Start the server:
   ```bash
   uvicorn main:app --reload
   ```

2. Access the API at:
   - Base URL: `http://127.0.0.1:8000`
   - Swagger UI: `http://127.0.0.1:8000/docs`
  
## How to Run the API (Prod)

1. Start the server:
   ```bash
   nohup uvicorn final:app --timeout-keep-alive 300 --host 0.0.0.0 --port (Enter Port Value, like 8000)
   ```

2. Access the API at:
   - Base URL: `your IP:Port`
   - Swagger UI: `your IP:Port/docs`
  
## How to Stop the API

1. Find PID for the Uvicorn instance:
   ```bash
   ps aux | grep uvicorn | grep -v grep
   ```
3. Kill the process:
   ```bash
   kill 0012345
   ```

## Directory Structure
```
<repository_directory>/
├── main.py           # Main API code
├── run_end_to_end.py # Inference and processing logic
├── cache_db.py       # Caching utilities
├── requirements.txt  # Dependency file
├── README.md         # Project documentation
└── ...               # Additional files
```

## Notes
- Ensure the `cache_files` directory exists and is writable.
- Adjust `allow_origins` in `CORS` middleware for production environments.
- Review and optimize `run_end_to_end` for your specific use case.

# Chess Score Sheet Digitization Pipeline

## Overview
This repository contains a Machine Learning (ML) pipeline for digitizing chess game score sheets. The pipeline processes scanned or photographed score sheets, extracts the moves, and converts them into a structured digital format. This project is designed to be deployed and managed using Kubernetes and is integrated with the [ChessGameManagement](https://github.com/benjaminkost/ChessGameManagement) repository (which is currently private).

## Features

## Experiments and Pipelines
- MLflow: https://dagshub.com/benjaminkost/chess-scoresheet-digitalizer.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D
- ZenML: https://cloud.zenml.io/workspaces/chesshub_zenml/projects

## Architecture
- MLflow
- ZenML

## Installation
### Prerequisites
- Python 3.12
- Docker

### Setup
1. Clone the repository:
   ```sh
   git clone [https://github.com/example/ChessScoreDigitization.git](https://github.com/benjaminkost/chess-scoresheet-digitalization.git)
   cd chess-scoresheet-digitalization
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Define .env
   copy ".env_sample" rename to ".env" and define variables that are missing

## Usage
1. Upload a scanned chess score sheet via the API or UI.
2. The pipeline processes the image and extracts moves.
3. Moves are validated and sent to ChessGameManagement.
4. Access the digitized game data through ChessGameManagement’s API.

## API Endpoints
| Endpoint        | Method | Description |
|---------------|--------|-------------|
| `/image/upload`      | POST   | Uploads a chess score sheet image |

## Integration with ChessGameManagement

## Contributing

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
For questions or support, open an issue.


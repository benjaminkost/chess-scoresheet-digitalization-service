# Chess-scoresheet-digitization-service
## Overview
This repository is part of a project called ChessHub. The ChessHub project is build with a Microservice-Architecture.

The chess-scoresheet-digitization-service repository defines the service which is responsible for digitializing a chess scoresheet.
## Installation
### Prerequisites
- Docker
- Python 3.12
### Setup
1. Clone the repository:
```bash
git clone https://github.com/benjaminkost/chess-scoresheet-digitalization-service.git
```
2. Define .env
- copy `.env_sample` and rename it to `.env`
- define the values according to your system
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Start project
- navigate to webapp and run
```bash
docker compose up -d
```

## API Endpoints

| Endpoint          | Method | Description                                      |
| ----------------- | ------ | ------------------------------------------------ |
| /api/image/upload | POST   | Upload image get a chess game as a PGN-File back |
## Test it out
 1. Open a browser and tip in: `http://localhost:8000/docs` 
 2. Try out the endpoints
## ML Experiments
- MLflow: https://dagshub.com/benjaminkost/chess-scoresheet-digitalizer.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D
- ZenML: https://cloud.zenml.io/workspaces/chesshub_zenml/projects
## Architecture
- ML experiments and Model logging: MLflow
- Pipeline Monitoring: ZenML
- Endpoints: RESTFull APIs
- Build and deploy: Docker compose
## Problems
Create an Issue in the Issues section in this repository.
## License
This project is licensed under the MIT License. See `License` for details.

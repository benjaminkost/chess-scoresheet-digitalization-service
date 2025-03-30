# Chess Score Sheet Digitization Pipeline

## Overview
This repository contains a Machine Learning (ML) pipeline for digitizing chess game score sheets. The pipeline processes scanned or photographed score sheets, extracts the moves, and converts them into a structured digital format. This project is designed to be deployed and managed using Kubernetes and is integrated with the [ChessGameManagement](https://github.com/benjaminkost/ChessGameManagement) repository (which is currently private).

## Features
- Handwritten Character Recognition (HCR) for extracting handwritten or printed chess notation
- Preprocessing techniques for image enhancement
- Machine Learning models for move recognition and validation
- Integration with ChessGameManagement for game storage and analysis
- Kubernetes deployment for scalability and reliability

## Architecture
1. **Image Processing**: Enhances the input images (noise reduction, thresholding, etc.).
2. **OCR & Move Extraction**: Uses ML-based OCR models to extract and interpret chess moves.
3. **Move Validation**: Ensures the extracted moves follow legal chess rules.
4. **Integration with ChessGameManagement**: Stores the digitized game data.
5. **Deployment via Kubernetes**: Provides containerized services for easy scalability.

## Installation
### Prerequisites
- Python 3.8+
- Docker & Kubernetes
- Helm (for managing Kubernetes deployments)
- OpenCV, TensorFlow/PyTorch, Tesseract OCR

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
3. Deploy to Kubernetes:
   ```sh
   kubectl apply -f k8s/
   ```

## Usage
1. Upload a scanned chess score sheet via the API or UI.
2. The pipeline processes the image and extracts moves.
3. Moves are validated and sent to ChessGameManagement.
4. Access the digitized game data through ChessGameManagement’s API.

## API Endpoints
| Endpoint        | Method | Description |
|---------------|--------|-------------|
| `/upload`      | POST   | Uploads a chess score sheet image |
| `/process`     | GET    | Starts processing the uploaded image |
| `/results`     | GET    | Fetches extracted and validated game moves |

## Kubernetes Deployment
- The repository includes Kubernetes manifests under the `k8s/` directory.
- Modify `values.yaml` to configure environment variables and resource limits.
- Use `helm upgrade --install chess-pipeline ./helm/` to deploy via Helm.

## Integration with ChessGameManagement
- The extracted moves are sent via REST API to ChessGameManagement.
- Ensure ChessGameManagement is running and accessible within the Kubernetes cluster.
- Update `config.yaml` with the correct ChessGameManagement API endpoint.

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit changes and push to your branch
4. Submit a Pull Request

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
For questions or support, open an issue.


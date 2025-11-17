# zk-backend

A minimal backend and tooling for running zero-knowledge (ZK) proofs and inference workflows used by EarthSkyOrg.

This repository contains smart-contract artifacts, Python services, example inputs, and helper scripts to build and run ZK verification and an example inference pipeline.

## Contents

- `contracts/` — Solidity source files, Foundry configuration, and contract tests.
- `srv/` — Python service, example inputs, model artifacts, and helper scripts for running inference and verification.
- `example/` — Example ONNX model and input used by the demo pipeline.
- `inputdata/`, `onnxmodel/`, `calibrate/` — Supporting data and calibration artifacts.

## Key files

- `srv/main.py` — Example Python entrypoint for running the inference/verification flow.
- `srv/requirements.txt` — Python dependencies for the service in `srv/`.
- `contracts/` — contains `Verifier.sol`, `Counter.sol`, and the Foundry `foundry.toml` and tests.
- `docker-compose.yml` — Compose file for running services (if applicable).

## Prerequisites

- Python 3.8+ (recommended) and `virtualenv` or `venv`.
- Foundry (for building and testing Solidity contracts) if you intend to work with `contracts/`.
- Docker & Docker Compose (optional) to run containerized services.

## Quickstart — Python service (local)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r srv\requirements.txt
```

3. Run the example service:

```powershell
python srv\main.py
```

The `srv/` directory includes `proof.json`, `Verifier.abi`, and an example `soccercircuit.compiled` file used by the demo flow.

## Quickstart — Contracts (Foundry)

If you want to build or test contracts in `contracts/` using Foundry:

```bash
cd contracts
forge test
```

Refer to the `contracts/README.md` for contract-specific setup and testing notes.

## Running with Docker Compose (optional)

If you prefer containers, you can start services with:

```powershell
docker-compose up --build
```

Adjust the compose configuration as needed for your environment.

## Repository layout

- `contracts/` — Solidity contracts, tests, and build config.
- `srv/` — Python service, model files, inputs, and example proof artifacts.
- `example/` — Example ONNX model and JSON input to exercise the pipeline.

## Contributing

Contributions are welcome. Please open issues for bugs or feature requests and submit pull requests with clear descriptions and tests where applicable.

## License & Contact

This repository does not include an explicit license file. If you intend to publish or share this code, add a `LICENSE` file describing the terms.

For questions or help, open an issue or contact the maintainers via the project repository.


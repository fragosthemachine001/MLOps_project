from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
import onnx
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from prometheus_client import CollectorRegistry, Counter, Histogram, Summary, make_asgi_app
from pydantic import BaseModel

from credit_card_fraud_analysis.hydra_config_loader import load_config
from credit_card_fraud_analysis.lightning_module import LitAutoEncoder
from credit_card_fraud_analysis.monitoring_utils import generate_drift_report, log_to_database
from credit_card_fraud_analysis.utils.my_logger import logger

try:
    import onnxruntime as ort
except ModuleNotFoundError:
    ort = None


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

MY_REGISTRY = CollectorRegistry()

error_counter = Counter("prediction_error", "Number of prediction errors", registry=MY_REGISTRY)
request_counter = Counter("prediction_requests", "Number of prediction requests", registry=MY_REGISTRY)
request_latency = Histogram("prediction_latency_seconds", "Prediction latency in seconds", registry=MY_REGISTRY)
# Using Summary to track the distribution of input feature lengths
feature_summary = Summary(
    "feature_length_summary", "Summary of the number of features in requests", registry=MY_REGISTRY
)


class TransactionRequest(BaseModel):
    features: List[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize both state variables
    app.state.model = None
    app.state.ort_session = None
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load PyTorch Model
    logger.info("Loading PyTorch Model")
    ckpt_files = list(MODELS_DIR.glob("*.ckpt"))
    if ckpt_files:
        latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
        try:
            app.state.model = LitAutoEncoder.load_from_checkpoint(latest_ckpt)
            app.state.model.to(app.state.device)
            app.state.model.eval()
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")

    target_model_path = MODELS_DIR / "optimized_model.onnx"
    if target_model_path.exists():
        logger.info(f"Priority: Found optimized model at {target_model_path.name}")
    else:
        onnx_files = list(MODELS_DIR.glob("*.onnx"))
        if onnx_files:
            target_model_path = max(onnx_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading most recent generic ONNX file: {target_model_path.name}")
        else:
            target_model_path = None
            logger.error("No ONNX files found in the models directory.")

    if target_model_path and ort is not None:
        try:
            model_str_path = str(target_model_path)

            # Initialize session
            app.state.ort_session = ort.InferenceSession(model_str_path)

            # Verification logic from class
            onnx_model = onnx.load(model_str_path)
            onnx.checker.check_model(onnx_model)

            logger.info("ONNX Model loaded and verified successfully.")
            # logger.debug(onnx.printer.to_text(onnx_model.graph))
        except Exception as e:
            logger.error(f"Failed to initialize ONNX session or checker: {e}")
            app.state.ort_session = None
    else:
        app.state.ort_session = None
        if target_model_path and ort is not None:
            logger.warning("ONNX model found but onnxruntime is not installed.")

    logger.info("Loading configurations")
    try:
        config = load_config()
        app.state.threshold = getattr(config.evaluation, "threshold_percentile", 0.005)
    except Exception:
        app.state.threshold = 0.005

    quant_path = MODELS_DIR / "optimized_model.onnx"
    if quant_path.exists() and ort is None:
        logger.info("Loading Quantized ONNX model for /predict_optimized")
        app.state.ort_session_quant = ort.InferenceSession(str(quant_path))
    else:
        app.state.ort_session_quant = None
        if quant_path.exists():
            logger.warning("Quantized model found but onnxruntime is not installed")
        else:
            logger.warning("Quantized model not found. /predict_optimized will fail.")

    yield
    # Cleanup
    if app.state.model:
        del app.state.model
    if app.state.ort_session:
        del app.state.ort_session
    if app.state.ort_session_quant:
        del app.state.ort_session_quant


app = FastAPI(title="Credit Card Fraud API", lifespan=lifespan)
app.mount("/metrics", make_asgi_app(registry=MY_REGISTRY))


@app.get("/")
def root():
    return {"message": "Credit Card Fraud Detection API is active"}


@app.post("/predict")
async def predict(
    request: Request, data: TransactionRequest, background_tasks: BackgroundTasks, use_onnx: bool = False
):
    # Track request count
    request_counter.inc()
    # Measure latency of the prediction process
    with request_latency.time():
        try:
            # Observe the input data complexity
            feature_summary.observe(len(data.features))

            expected_dim = 28
            if len(data.features) != expected_dim:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dimension mismatch: expected {expected_dim} features, got {len(data.features)}",
                )

            threshold = request.app.state.threshold
            mse_loss = 0

            if use_onnx:
                if ort is None:
                    raise HTTPException(
                        status_code=500, detail="onnxrutime is not installed. Install it to enable ONNX iference."
                    )

                session = request.app.state.ort_session
                if not session:
                    raise HTTPException(status_code=503, detail="ONNX session not available")

                input_data = np.array([data.features], dtype=np.float32)
                input_name = session.get_inputs()[0].name
                outputs = session.run(None, {input_name: input_data})
                mse_loss = np.mean((input_data - outputs[0]) ** 2)
            else:
                model = request.app.state.model
                if not model:
                    raise HTTPException(status_code=503, detail="PyTorch model not available")

                input_tensor = torch.tensor([data.features], dtype=torch.float32).to(request.app.state.device)
                with torch.no_grad():
                    reconstruction = model(input_tensor)
                    mse_loss = torch.mean((input_tensor - reconstruction) ** 2).item()

            is_fraud = bool(mse_loss > request.app.state.threshold)
            background_tasks.add_task(log_to_database, data.features, float(mse_loss), is_fraud)

            return {
                "is_fraud": bool(mse_loss > threshold),
                "reconstruction_error": float(mse_loss),
                "engine": "onnx" if use_onnx else "pytorch",
            }

        except Exception as e:
            # Increment error count on any exception
            error_counter.inc()
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring")
def monitoring():
    reference_path = MODELS_DIR / "reference_data.csv"
    json_str = generate_drift_report(reference_path)
    return json_str


@app.post("/predict_optimized")
async def predict_optimized(data: TransactionRequest, background_tasks: BackgroundTasks):
    if ort is None or app.state.ort_session_quant is None:
        raise HTTPException(status_code=503, detail="Quantized ONNX session not available.")

    try:
        session = app.state.ort_session_quant
        input_data = np.array([data.features], dtype=np.float32)
        input_name = session.get_inputs()[0].name

        with request_latency.time():
            outputs = session.run(None, {input_name: input_data})
            reconstruction = outputs[0]
            mse_loss = np.mean((input_data - reconstruction) ** 2)

        is_fraud = bool(mse_loss > app.state.threshold)

        background_tasks.add_task(log_to_database, data.features, float(mse_loss), is_fraud)

        return {
            "is_fraud": is_fraud,
            "reconstruction_error": float(mse_loss),
            "engine": "onnx_quantized_8bit",
            "optimization_techniques": ["pruning", "quantization"],
        }
    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e))

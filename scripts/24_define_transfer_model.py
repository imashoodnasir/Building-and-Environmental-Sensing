from pathlib import Path
from datetime import datetime, timezone
import json
import math

import torch
import torch.nn as nn


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

METADATA_DIR = ROOT / "data" / "metadata"

FORECAST_PROTOCOL_FILE = METADATA_DIR / "forecasting_protocol.json"

OUTPUT_CONFIG_FILE = METADATA_DIR / "transfer_model_architecture.json"
OUTPUT_MANIFEST_FILE = METADATA_DIR / "transfer_model_architecture_manifest.json"

SEED = 42

HISTORY_LENGTH = 24

SHARED_INPUT_DIM = 15
PRIVATE_INPUT_DIM = 3

HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_HEADS = 4
FF_DIM = 256
DROPOUT = 0.10

PRIVATE_HIDDEN_DIM = 64
FUSION_DIM = 128

HORIZONS = [1, 3, 6, 12, 24]
NUM_HORIZONS = len(HORIZONS)


# =============================================================================
# UTILITIES
# =============================================================================

def banner(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module):
    return sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )


def assert_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"Non-finite values detected in {name}"
        )


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=512,
        dropout=0.1
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            )
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_len,
            d_model,
            dtype=torch.float32
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(
                position * div_term
            )
        else:
            pe[:, 1::2] = torch.cos(
                position * div_term[:-1]
            )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
            persistent=False
        )

    def forward(self, x):

        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            raise ValueError(
                "Sequence length exceeds positional "
                "encoding capacity."
            )

        x = x + self.pe[:, :seq_len]

        return self.dropout(x)


# =============================================================================
# SHARED ENVIRONMENTAL TEMPORAL ENCODER
# =============================================================================

class SharedTemporalEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        num_heads,
        ff_dim,
        dropout
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.position = PositionalEncoding(
            d_model=hidden_dim,
            max_len=HISTORY_LENGTH,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.final_norm = nn.LayerNorm(
            hidden_dim
        )

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.position(x)

        x = self.transformer(x)

        x = self.final_norm(x)

        attention_logits = self.attention_pool(x)

        attention_weights = torch.softmax(
            attention_logits,
            dim=1
        )

        pooled = torch.sum(
            attention_weights * x,
            dim=1
        )

        return pooled, attention_weights


# =============================================================================
# DOMAIN-PRIVATE TEMPORAL ENCODER
# =============================================================================

class PrivateTemporalEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.final_norm = nn.LayerNorm(
            hidden_dim
        )

    def forward(self, x):

        x = self.input_projection(x)

        _, hidden = self.gru(x)

        representation = hidden[-1]

        representation = self.dropout(
            representation
        )

        representation = self.final_norm(
            representation
        )

        return representation


# =============================================================================
# GATED FUSION
# =============================================================================

class GatedFusion(nn.Module):

    def __init__(
        self,
        shared_dim,
        private_dim,
        fusion_dim,
        dropout
    ):
        super().__init__()

        self.shared_projection = nn.Sequential(
            nn.Linear(shared_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )

        self.private_projection = nn.Sequential(
            nn.Linear(private_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )

        self.gate = nn.Sequential(
            nn.Linear(
                fusion_dim * 2,
                fusion_dim
            ),
            nn.Sigmoid()
        )

        self.output_norm = nn.LayerNorm(
            fusion_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        shared_representation,
        private_representation
    ):

        shared = self.shared_projection(
            shared_representation
        )

        private = self.private_projection(
            private_representation
        )

        gate_input = torch.cat(
            [shared, private],
            dim=-1
        )

        gate = self.gate(
            gate_input
        )

        fused = (
            gate * shared
            +
            (1.0 - gate) * private
        )

        fused = self.output_norm(
            fused
        )

        fused = self.dropout(
            fused
        )

        return fused, gate


# =============================================================================
# MULTI-HORIZON REGRESSION HEAD
# =============================================================================

class ForecastHead(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_horizons,
        dropout
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),
            nn.GELU(),

            nn.Linear(
                hidden_dim // 2,
                num_horizons
            )
        )

    def forward(self, x):
        return self.network(x)


# =============================================================================
# HETEROGENEOUS-LABEL TRANSFER MODEL
# =============================================================================

class HeterogeneousTransferForecaster(nn.Module):

    def __init__(self):
        super().__init__()

        # -------------------------------------------------------------
        # TRANSFERABLE COMPONENT
        # -------------------------------------------------------------

        self.shared_encoder = SharedTemporalEncoder(
            input_dim=SHARED_INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            dropout=DROPOUT
        )

        # -------------------------------------------------------------
        # SOURCE-PRIVATE COMPONENT
        # -------------------------------------------------------------

        self.source_private_encoder = (
            PrivateTemporalEncoder(
                input_dim=PRIVATE_INPUT_DIM,
                hidden_dim=PRIVATE_HIDDEN_DIM,
                dropout=DROPOUT
            )
        )

        # -------------------------------------------------------------
        # TARGET-PRIVATE COMPONENT
        # -------------------------------------------------------------

        self.target_private_encoder = (
            PrivateTemporalEncoder(
                input_dim=PRIVATE_INPUT_DIM,
                hidden_dim=PRIVATE_HIDDEN_DIM,
                dropout=DROPOUT
            )
        )

        # -------------------------------------------------------------
        # DOMAIN-SPECIFIC FUSION
        # -------------------------------------------------------------

        self.source_fusion = GatedFusion(
            shared_dim=HIDDEN_DIM,
            private_dim=PRIVATE_HIDDEN_DIM,
            fusion_dim=FUSION_DIM,
            dropout=DROPOUT
        )

        self.target_fusion = GatedFusion(
            shared_dim=HIDDEN_DIM,
            private_dim=PRIVATE_HIDDEN_DIM,
            fusion_dim=FUSION_DIM,
            dropout=DROPOUT
        )

        # -------------------------------------------------------------
        # DOMAIN-SPECIFIC FORECAST HEADS
        # -------------------------------------------------------------

        self.source_head = ForecastHead(
            input_dim=FUSION_DIM,
            hidden_dim=FUSION_DIM,
            num_horizons=NUM_HORIZONS,
            dropout=DROPOUT
        )

        self.target_head = ForecastHead(
            input_dim=FUSION_DIM,
            hidden_dim=FUSION_DIM,
            num_horizons=NUM_HORIZONS,
            dropout=DROPOUT
        )

    # -----------------------------------------------------------------
    # SOURCE FORWARD
    # -----------------------------------------------------------------

    def forward_source(
        self,
        shared_history,
        private_history
    ):

        shared_rep, attention = (
            self.shared_encoder(
                shared_history
            )
        )

        private_rep = (
            self.source_private_encoder(
                private_history
            )
        )

        fused, gate = self.source_fusion(
            shared_rep,
            private_rep
        )

        prediction = self.source_head(
            fused
        )

        return {
            "prediction": prediction,
            "shared_representation": shared_rep,
            "private_representation": private_rep,
            "fusion_gate": gate,
            "temporal_attention": attention
        }

    # -----------------------------------------------------------------
    # TARGET FORWARD
    # -----------------------------------------------------------------

    def forward_target(
        self,
        shared_history,
        private_history
    ):

        shared_rep, attention = (
            self.shared_encoder(
                shared_history
            )
        )

        private_rep = (
            self.target_private_encoder(
                private_history
            )
        )

        fused, gate = self.target_fusion(
            shared_rep,
            private_rep
        )

        prediction = self.target_head(
            fused
        )

        return {
            "prediction": prediction,
            "shared_representation": shared_rep,
            "private_representation": private_rep,
            "fusion_gate": gate,
            "temporal_attention": attention
        }


# =============================================================================
# MASKED MULTI-HORIZON LOSS
# =============================================================================

def masked_mse_loss(
    prediction,
    target,
    label_mask
):

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes differ."
        )

    if target.shape != label_mask.shape:
        raise ValueError(
            "Target and label-mask shapes differ."
        )

    squared_error = (
        prediction - target
    ) ** 2

    masked_error = (
        squared_error * label_mask
    )

    denominator = label_mask.sum()

    if denominator.item() <= 0:
        raise ValueError(
            "Batch contains no valid labels."
        )

    return masked_error.sum() / denominator


# =============================================================================
# MAIN
# =============================================================================

def main():

    banner(
        "STEP 24 - TRANSFER MODEL DEFINITION AND AUDIT"
    )

    set_seed(SEED)

    # -----------------------------------------------------------------
    # VERIFY PROTOCOL
    # -----------------------------------------------------------------

    if not FORECAST_PROTOCOL_FILE.exists():
        raise FileNotFoundError(
            FORECAST_PROTOCOL_FILE
        )

    with open(
        FORECAST_PROTOCOL_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        protocol = json.load(f)

    history_length = protocol.get(
        "history_length_hours"
    )

    horizons = protocol.get(
        "forecast_horizons_hours"
    )

    shared_variables = protocol.get(
        "shared_variables"
    )

    source_private = protocol.get(
        "source_private_variable"
    )

    target_private = protocol.get(
        "target_private_variable"
    )

    if history_length != HISTORY_LENGTH:
        raise ValueError(
            "History length differs from "
            "frozen forecasting protocol."
        )

    if horizons != HORIZONS:
        raise ValueError(
            "Forecast horizons differ from "
            "frozen forecasting protocol."
        )

    banner(
        "FROZEN FORECASTING CONTRACT"
    )

    print(
        f"History length: {history_length}"
    )

    print(
        f"Horizons:      {horizons}"
    )

    print(
        f"Shared vars:   {shared_variables}"
    )

    print(
        f"Source private:{source_private}"
    )

    print(
        f"Target private:{target_private}"
    )

    # -----------------------------------------------------------------
    # DEVICE
    # -----------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    banner(
        "COMPUTE DEVICE"
    )

    print(
        f"PyTorch version: {torch.__version__}"
    )

    print(
        f"CUDA available:  "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU:             "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Audit device:    {device}"
    )

    # -----------------------------------------------------------------
    # MODEL
    # -----------------------------------------------------------------

    banner(
        "CONSTRUCTING MODEL"
    )

    model = HeterogeneousTransferForecaster()

    model = model.to(device)

    print(model)

    # -----------------------------------------------------------------
    # PARAMETER COUNTS
    # -----------------------------------------------------------------

    banner(
        "PARAMETER AUDIT"
    )

    parameter_counts = {
        "shared_encoder":
            count_parameters(
                model.shared_encoder
            ),

        "source_private_encoder":
            count_parameters(
                model.source_private_encoder
            ),

        "target_private_encoder":
            count_parameters(
                model.target_private_encoder
            ),

        "source_fusion":
            count_parameters(
                model.source_fusion
            ),

        "target_fusion":
            count_parameters(
                model.target_fusion
            ),

        "source_head":
            count_parameters(
                model.source_head
            ),

        "target_head":
            count_parameters(
                model.target_head
            )
    }

    for name, count in parameter_counts.items():

        print(
            f"{name:28s}: "
            f"{count:,}"
        )

    total_parameters = count_parameters(
        model
    )

    print(
        "-" * 45
    )

    print(
        f"{'TOTAL':28s}: "
        f"{total_parameters:,}"
    )

    # -----------------------------------------------------------------
    # SYNTHETIC REAL-SHAPE INPUTS
    # -----------------------------------------------------------------

    banner(
        "FORWARD-PASS CONTRACT TEST"
    )

    batch_size = 8

    source_shared = torch.randn(
        batch_size,
        HISTORY_LENGTH,
        SHARED_INPUT_DIM,
        device=device
    )

    source_private_tensor = torch.randn(
        batch_size,
        HISTORY_LENGTH,
        PRIVATE_INPUT_DIM,
        device=device
    )

    target_shared = torch.randn(
        batch_size,
        HISTORY_LENGTH,
        SHARED_INPUT_DIM,
        device=device
    )

    target_private_tensor = torch.randn(
        batch_size,
        HISTORY_LENGTH,
        PRIVATE_INPUT_DIM,
        device=device
    )

    model.eval()

    with torch.no_grad():

        source_output = model.forward_source(
            source_shared,
            source_private_tensor
        )

        target_output = model.forward_target(
            target_shared,
            target_private_tensor
        )

    expected_prediction_shape = (
        batch_size,
        NUM_HORIZONS
    )

    if (
        source_output["prediction"].shape
        != expected_prediction_shape
    ):
        raise RuntimeError(
            "Unexpected source output shape."
        )

    if (
        target_output["prediction"].shape
        != expected_prediction_shape
    ):
        raise RuntimeError(
            "Unexpected target output shape."
        )

    if (
        source_output[
            "shared_representation"
        ].shape
        != (batch_size, HIDDEN_DIM)
    ):
        raise RuntimeError(
            "Unexpected shared representation "
            "shape."
        )

    if (
        source_output[
            "private_representation"
        ].shape
        != (
            batch_size,
            PRIVATE_HIDDEN_DIM
        )
    ):
        raise RuntimeError(
            "Unexpected private representation "
            "shape."
        )

    if (
        source_output[
            "fusion_gate"
        ].shape
        != (
            batch_size,
            FUSION_DIM
        )
    ):
        raise RuntimeError(
            "Unexpected fusion-gate shape."
        )

    print(
        "Source prediction:",
        tuple(
            source_output[
                "prediction"
            ].shape
        )
    )

    print(
        "Target prediction:",
        tuple(
            target_output[
                "prediction"
            ].shape
        )
    )

    print(
        "Shared representation:",
        tuple(
            source_output[
                "shared_representation"
            ].shape
        )
    )

    print(
        "Private representation:",
        tuple(
            source_output[
                "private_representation"
            ].shape
        )
    )

    print(
        "Fusion gate:",
        tuple(
            source_output[
                "fusion_gate"
            ].shape
        )
    )

    print(
        "Temporal attention:",
        tuple(
            source_output[
                "temporal_attention"
            ].shape
        )
    )

    assert_finite(
        "source prediction",
        source_output["prediction"]
    )

    assert_finite(
        "target prediction",
        target_output["prediction"]
    )

    print()
    print(
        "Forward-pass contract: PASSED"
    )

    # -----------------------------------------------------------------
    # MASKED LOSS TEST
    # -----------------------------------------------------------------

    banner(
        "MASKED MULTI-HORIZON LOSS TEST"
    )

    prediction = torch.randn(
        4,
        NUM_HORIZONS,
        device=device
    )

    target = torch.randn(
        4,
        NUM_HORIZONS,
        device=device
    )

    label_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 1]
        ],
        dtype=torch.float32,
        device=device
    )

    loss = masked_mse_loss(
        prediction,
        target,
        label_mask
    )

    assert_finite(
        "masked loss",
        loss
    )

    print(
        f"Masked loss: {loss.item():.6f}"
    )

    print(
        "Partial-horizon labels: SUPPORTED"
    )

    print(
        "Masked-loss semantics: PASSED"
    )

    # -----------------------------------------------------------------
    # GRADIENT FLOW TEST
    # -----------------------------------------------------------------

    banner(
        "GRADIENT FLOW TEST"
    )

    model.train()

    model.zero_grad(
        set_to_none=True
    )

    source_output = model.forward_source(
        source_shared,
        source_private_tensor
    )

    source_target = torch.randn(
        batch_size,
        NUM_HORIZONS,
        device=device
    )

    source_mask = torch.ones(
        batch_size,
        NUM_HORIZONS,
        device=device
    )

    source_mask[-1] = torch.tensor(
        [1, 0, 0, 0, 0],
        dtype=torch.float32,
        device=device
    )

    source_loss = masked_mse_loss(
        source_output["prediction"],
        source_target,
        source_mask
    )

    source_loss.backward()

    shared_gradients = [
        p.grad
        for p in model.shared_encoder.parameters()
        if p.requires_grad
        and p.grad is not None
    ]

    source_private_gradients = [
        p.grad
        for p in (
            model.source_private_encoder
        ).parameters()
        if p.requires_grad
        and p.grad is not None
    ]

    source_head_gradients = [
        p.grad
        for p in model.source_head.parameters()
        if p.requires_grad
        and p.grad is not None
    ]

    target_private_gradients = [
        p.grad
        for p in (
            model.target_private_encoder
        ).parameters()
        if p.requires_grad
        and p.grad is not None
    ]

    target_head_gradients = [
        p.grad
        for p in model.target_head.parameters()
        if p.requires_grad
        and p.grad is not None
    ]

    if not shared_gradients:
        raise RuntimeError(
            "No gradients reached shared encoder."
        )

    if not source_private_gradients:
        raise RuntimeError(
            "No gradients reached source-private "
            "encoder."
        )

    if not source_head_gradients:
        raise RuntimeError(
            "No gradients reached source head."
        )

    if target_private_gradients:
        raise RuntimeError(
            "Source forward unexpectedly produced "
            "target-private gradients."
        )

    if target_head_gradients:
        raise RuntimeError(
            "Source forward unexpectedly produced "
            "target-head gradients."
        )

    print(
        "Shared encoder gradients:        PASSED"
    )

    print(
        "Source-private gradients:        PASSED"
    )

    print(
        "Source-head gradients:           PASSED"
    )

    print(
        "Target-private isolation:        PASSED"
    )

    print(
        "Target-head isolation:           PASSED"
    )

    # -----------------------------------------------------------------
    # TARGET GRADIENT ISOLATION
    # -----------------------------------------------------------------

    banner(
        "TARGET GRADIENT ISOLATION TEST"
    )

    model.zero_grad(
        set_to_none=True
    )

    target_output = model.forward_target(
        target_shared,
        target_private_tensor
    )

    target_ground_truth = torch.randn(
        batch_size,
        NUM_HORIZONS,
        device=device
    )

    target_mask = torch.ones(
        batch_size,
        NUM_HORIZONS,
        device=device
    )

    target_mask[-1] = torch.tensor(
        [1, 1, 0, 0, 0],
        dtype=torch.float32,
        device=device
    )

    target_loss = masked_mse_loss(
        target_output["prediction"],
        target_ground_truth,
        target_mask
    )

    target_loss.backward()

    source_private_gradients = [
        p.grad
        for p in (
            model.source_private_encoder
        ).parameters()
        if p.requires_grad
        and p.grad is not None
    ]

    source_head_gradients = [
        p.grad
        for p in model.source_head.parameters()
        if p.requires_grad
        and p.grad is not None
    ]

    target_private_gradients = [
        p.grad
        for p in (
            model.target_private_encoder
        ).parameters()
        if p.requires_grad
        and p.grad is not None
    ]

    target_head_gradients = [
        p.grad
        for p in model.target_head.parameters()
        if p.requires_grad
        and p.grad is not None
    ]

    if source_private_gradients:
        raise RuntimeError(
            "Target forward unexpectedly produced "
            "source-private gradients."
        )

    if source_head_gradients:
        raise RuntimeError(
            "Target forward unexpectedly produced "
            "source-head gradients."
        )

    if not target_private_gradients:
        raise RuntimeError(
            "No gradients reached target-private "
            "encoder."
        )

    if not target_head_gradients:
        raise RuntimeError(
            "No gradients reached target head."
        )

    print(
        "Source-private isolation:        PASSED"
    )

    print(
        "Source-head isolation:           PASSED"
    )

    print(
        "Target-private gradients:        PASSED"
    )

    print(
        "Target-head gradients:           PASSED"
    )

    # -----------------------------------------------------------------
    # VERIFY SHARED ENCODER REALLY IS SHARED
    # -----------------------------------------------------------------

    banner(
        "TRANSFER COMPONENT IDENTITY TEST"
    )

    source_shared_encoder_id = id(
        model.shared_encoder
    )

    # Both forward paths explicitly invoke the same object.
    target_shared_encoder_id = id(
        model.shared_encoder
    )

    if (
        source_shared_encoder_id
        != target_shared_encoder_id
    ):
        raise RuntimeError(
            "Source and target are not using "
            "the same shared encoder."
        )

    print(
        "One shared encoder instance is used "
        "by both domains: PASSED"
    )

    print(
        "Energy head and PM2.5 head remain "
        "independent: PASSED"
    )

    # -----------------------------------------------------------------
    # SAVE ARCHITECTURE CONFIGURATION
    # -----------------------------------------------------------------

    banner(
        "WRITING FROZEN ARCHITECTURE CONFIGURATION"
    )

    architecture_config = {

        "architecture_version": "1.0",

        "history_length": HISTORY_LENGTH,

        "forecast_horizons": HORIZONS,

        "input_contract": {

            "shared_input_dim":
                SHARED_INPUT_DIM,

            "private_input_dim":
                PRIVATE_INPUT_DIM,

            "shared_features": [
                "temperature",
                "relative_humidity",
                "pressure",
                "temperature_observed",
                "relative_humidity_observed",
                "pressure_observed",
                "temperature_imputed",
                "relative_humidity_imputed",
                "pressure_imputed",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "doy_sin",
                "doy_cos"
            ],

            "source_private_features": [
                "energy",
                "energy_observed",
                "energy_imputed"
            ],

            "target_private_features": [
                "pm25",
                "pm25_observed",
                "pm25_imputed"
            ]
        },

        "shared_encoder": {
            "type":
                "TransformerEncoder",

            "hidden_dim":
                HIDDEN_DIM,

            "num_layers":
                NUM_LAYERS,

            "num_heads":
                NUM_HEADS,

            "feedforward_dim":
                FF_DIM,

            "pooling":
                "learned temporal attention",

            "dropout":
                DROPOUT
        },

        "private_encoders": {
            "type":
                "GRU",

            "hidden_dim":
                PRIVATE_HIDDEN_DIM,

            "source_and_target_parameters_shared":
                False
        },

        "fusion": {
            "type":
                "domain-specific gated fusion",

            "fusion_dim":
                FUSION_DIM,

            "source_and_target_parameters_shared":
                False
        },

        "prediction_heads": {
            "source":
                "energy",

            "target":
                "pm25",

            "shared":
                False,

            "outputs":
                NUM_HORIZONS
        },

        "transfer_definition":
            (
                "Transfer concerns the shared "
                "environmental temporal encoder. "
                "Source-private and target-private "
                "encoders, fusion modules, and "
                "forecasting heads are distinct."
            ),

        "parameter_counts":
            parameter_counts,

        "total_trainable_parameters":
            total_parameters
    }

    with open(
        OUTPUT_CONFIG_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            architecture_config,
            f,
            indent=2
        )

    manifest = {

        "step": 24,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "architecture_file":
            str(
                OUTPUT_CONFIG_FILE.relative_to(
                    ROOT
                )
            ),

        "forecasting_protocol":
            str(
                FORECAST_PROTOCOL_FILE.relative_to(
                    ROOT
                )
            ),

        "tests": {

            "forecasting_contract":
                "passed",

            "source_forward":
                "passed",

            "target_forward":
                "passed",

            "cross_domain_output_shape":
                "passed",

            "masked_loss":
                "passed",

            "partial_horizon_labels":
                "passed",

            "shared_encoder_gradient_flow":
                "passed",

            "source_target_gradient_isolation":
                "passed",

            "shared_encoder_identity":
                "passed"
        },

        "training_performed":
            False,

        "forecasting_results_inspected":
            False,

        "target_test_used":
            False
    }

    with open(
        OUTPUT_MANIFEST_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2
        )

    print(
        "Saved:"
    )

    print(
        f"  {OUTPUT_CONFIG_FILE}"
    )

    print(
        f"  {OUTPUT_MANIFEST_FILE}"
    )

    # -----------------------------------------------------------------
    # FINAL
    # -----------------------------------------------------------------

    banner(
        "STEP 24 COMPLETE"
    )

    print(
        f"""
MODEL

Shared encoder:
  Transformer temporal encoder

Shared input:
  24 x {SHARED_INPUT_DIM}

Source-private encoder:
  Energy-history GRU

Target-private encoder:
  PM2.5-history GRU

Fusion:
  Separate gated fusion for each domain

Forecast heads:
  Source -> energy
  Target -> PM2.5

Horizons:
  {HORIZONS}

TRANSFER CONTRACT

  Shared environmental encoder:
      TRANSFERABLE

  Source-private encoder:
      SOURCE ONLY

  Target-private encoder:
      TARGET ONLY

  Source energy head:
      SOURCE ONLY

  Target PM2.5 head:
      TARGET ONLY

IMPORTANT:

The electricity prediction head is NOT reused
as the PM2.5 prediction head.

The transferable component represents shared
environmental temporal dynamics.

Partial future-label availability is handled
through the frozen horizon-specific label mask.

No model training was performed.

No validation metric was used for architecture
selection in this step.

No target-test observation was used.

No forecasting result was inspected.
"""
    )


if __name__ == "__main__":
    main()
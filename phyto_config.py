"""
PhytoSentinel-AESTIN Configuration
All hyperparameters in one place. Modify here before running experiments.
"""

# ── Dataset / Simulation ──────────────────────────────────────────────────────
NUM_NODES       = 100       # number of crop-field nodes in the graph
NUM_TIMESTEPS   = 30        # simulation time steps per sample
NUM_GRAPHS      = 500       # total graph instances in dataset
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
# TEST_SPLIT is the remainder

NODE_FEAT_DIM   = 8         # per-node feature dimensions (crop health indicators)
EDGE_FEAT_DIM   = 4         # per-edge met features: [wind_speed, humidity, wind_alignment, distance]
NUM_CLASSES     = 2         # binary: infected / not-infected within the horizon

# Prediction horizon: predict whether a currently-Susceptible node becomes infected
# within the NEXT K steps. K>1 forces multi-hop reasoning — infection can arrive via
# 2-3 hop paths that a 1-hop heuristic cannot see — which is where message-passing
# GNNs earn their keep. K=3 matches the 3-layer GNN receptive field.
HORIZON         = 3

RANDOM_SEED     = 42

# ── DAGCA (Dynamic Atmospheric Graph Construction Algorithm) ──────────────────
DAGCA_HIDDEN    = 64        # hidden units in the viability MLP
DAGCA_LAYERS    = 2         # MLP depth
DAGCA_DROPOUT   = 0.1
EDGE_THRESHOLD  = 0.3       # edges with weight < threshold are pruned at inference

# ── Bayesian DAGCA ─────────────────────────────────────────────────────────────
BAYESIAN_SAMPLES = 5        # Monte Carlo samples per forward pass during training
PRIOR_ALPHA     = 1.0       # Beta prior α
PRIOR_BETA      = 1.0       # Beta prior β
KL_WEIGHT       = 1e-3      # weight of KL divergence in loss

# ── GNN (Message Passing Network) ────────────────────────────────────────────
GNN_HIDDEN      = 128
GNN_LAYERS      = 3
GNN_DROPOUT     = 0.3
GNN_TYPE        = "sage"    # "sage" | "gat" | "gcn"
GAT_HEADS       = 4

# ── SENR0 (Spectral Epidemic Network R0) ─────────────────────────────────────
SENR0_GAMMA     = 0.1       # recovery rate (used in R0 = lambda_max / gamma)

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS          = 100
BATCH_SIZE      = 32
LR              = 3e-4
WEIGHT_DECAY    = 1e-5
SCHEDULER       = "cosine"  # "cosine" | "step" | "none"
PATIENCE        = 20        # early stopping patience

# ── Logging / Saving ──────────────────────────────────────────────────────────
CHECKPOINT_DIR  = "results/checkpoints"
RESULTS_DIR     = "results"
LOG_INTERVAL    = 10        # log every N epochs

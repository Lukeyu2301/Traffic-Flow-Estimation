import os
import gc
import random

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb

from torch_geometric.nn import GATv2Conv, BatchNorm
from torch_geometric.utils import negative_sampling, to_undirected

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. 基础配置
# ============================================================

SEED = 42


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print(f">>> Device: {device}")


# ============================================================
# 2. 时段与路径设置
# ============================================================

# 修改这里即可运行 morning / noon / evening
PERIOD = "evening"

road_path = (
    r'D:\KEKE\宣城交通流量预测\data\road_final_features.csv'
)

traffic_path = (
    rf'D:\KEKE\宣城交通流量预测\data\{PERIOD.capitalize()}_Average.csv'
)

save_dir = (
    rf'results\interp\{PERIOD}\seed_{SEED}'
)

os.makedirs(
    save_dir,
    exist_ok=True
)


# ============================================================
# 3. GAT Encoder
# ============================================================

class GATEncoder(nn.Module):

    def __init__(
        self,
        in_c,
        hid_c,
        out_c,
        num_layers,
        heads,
        dropout
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # ----------------------------------------------------
        # 第一层
        # ----------------------------------------------------

        self.convs.append(
            GATv2Conv(
                in_c,
                hid_c,
                heads=heads,
                edge_dim=1,
                dropout=dropout
            )
        )

        self.bns.append(
            BatchNorm(
                hid_c * heads
            )
        )

        # ----------------------------------------------------
        # 中间层
        # ----------------------------------------------------

        for _ in range(
            num_layers - 2
        ):

            self.convs.append(
                GATv2Conv(
                    hid_c * heads,
                    hid_c,
                    heads=heads,
                    edge_dim=1,
                    dropout=dropout
                )
            )

            self.bns.append(
                BatchNorm(
                    hid_c * heads
                )
            )

        # ----------------------------------------------------
        # 最后一层：
        # heads固定为1，输出维度为 out_c
        # ----------------------------------------------------

        self.convs.append(
            GATv2Conv(
                hid_c * heads,
                out_c,
                heads=1,
                edge_dim=1,
                dropout=dropout
            )
        )

        # ----------------------------------------------------
        # Contrastive projection head
        # ----------------------------------------------------

        self.proj = nn.Sequential(
            nn.Linear(
                out_c,
                out_c
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                out_c,
                out_c
            )
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr
    ):

        for i in range(
            len(self.convs) - 1
        ):

            x = self.convs[i](
                x,
                edge_index,
                edge_attr
            )

            x = self.bns[i](x)

            x = F.elu(x)

        # ----------------------------------------------------
        # pre-projection embedding
        # ----------------------------------------------------

        z = self.convs[-1](
            x,
            edge_index,
            edge_attr
        )

        # ----------------------------------------------------
        # projection for contrastive training
        # ----------------------------------------------------

        p = self.proj(z)

        return z, p


# ============================================================
# 4. GAT-MLP baseline
# ============================================================

class GATMLPModel(nn.Module):

    def __init__(
        self,
        in_c,
        hidden_channels,
        embedding_dim,
        num_layers,
        heads,
        dropout
    ):
        super().__init__()

        self.encoder = GATEncoder(
            in_c=in_c,
            hid_c=hidden_channels,
            out_c=embedding_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        )

        self.regressor = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_channels
            ),

            nn.LayerNorm(
                hidden_channels
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_channels,
                1
            )
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr
    ):

        z, _ = self.encoder(
            x,
            edge_index,
            edge_attr
        )

        return self.regressor(z)


# ============================================================
# 5. 指标函数
# ============================================================

def compute_all_metrics(
    y_true_sc,
    y_pred_sc,
    scaler_y
):

    # --------------------------------------------------------
    # inverse standardization
    # --------------------------------------------------------

    y_true_log = (
        scaler_y
        .inverse_transform(
            np.asarray(
                y_true_sc
            ).reshape(-1, 1)
        )
        .flatten()
    )

    y_pred_log = (
        scaler_y
        .inverse_transform(
            np.asarray(
                y_pred_sc
            ).reshape(-1, 1)
        )
        .flatten()
    )

    # --------------------------------------------------------
    # inverse log1p
    # --------------------------------------------------------

    y_true = np.expm1(
        y_true_log
    )

    y_pred = np.expm1(
        y_pred_log
    )

    y_pred = np.maximum(
        y_pred,
        0
    )

    # --------------------------------------------------------
    # metrics
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    mask = y_true > 0

    mape = (
        np.mean(
            np.abs(
                (
                    y_true[mask]
                    -
                    y_pred[mask]
                )
                /
                y_true[mask]
            )
        )
        *
        100
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape
    }


# ============================================================
# 6. InfoNCE-based pairwise contrastive loss
# ============================================================

def contrastive_loss(
    projection,
    edge_index,
    temperature,
    num_nodes
):

    # --------------------------------------------------------
    # Normalize projected representations
    # --------------------------------------------------------

    p_n = F.normalize(
        projection,
        p=2,
        dim=1
    )

    # --------------------------------------------------------
    # Positive pairs:
    # connected road segments
    # --------------------------------------------------------

    pos_score = torch.sum(
        p_n[edge_index[0]]
        *
        p_n[edge_index[1]],
        dim=-1
    )

    pos_score = (
        pos_score
        /
        temperature
    )

    # --------------------------------------------------------
    # Negative pairs:
    # randomly sampled non-connected road segments
    # --------------------------------------------------------

    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=edge_index.size(1)
    )

    neg_score = torch.sum(
        p_n[neg_edge_index[0]]
        *
        p_n[neg_edge_index[1]],
        dim=-1
    )

    neg_score = (
        neg_score
        /
        temperature
    )

    # --------------------------------------------------------
    # Equivalent to:
    #
    # -log(
    # exp(pos) /
    # [exp(pos)+exp(neg)]
    # )
    #
    # but numerically more stable
    # --------------------------------------------------------

    loss = F.softplus(
        neg_score
        -
        pos_score
    ).mean()

    return loss


# ============================================================
# 7. 数据加载与预处理
# ============================================================

print(
    "\n>>> 加载数据..."
)

df_raw = pd.read_csv(
    road_path,
    encoding='gbk'
)

df_traffic = pd.read_csv(
    traffic_path
)

data_all = df_raw.merge(
    df_traffic,
    left_on='cid',
    right_on='ROAD_ID'
)


# ------------------------------------------------------------
# Feature columns
# ------------------------------------------------------------

exclude = [
    'cid',
    'ROAD_ID',
    'avg_flow',
    'geometry',
    'ROAD_NAME',
    'Unnamed: 0'
]

feature_cols = [
    c
    for c in data_all.columns
    if c not in exclude
]


# ------------------------------------------------------------
# Encode road class
# ------------------------------------------------------------

if 'fclass' in data_all.columns:

    data_all['fclass'] = (
        LabelEncoder()
        .fit_transform(
            data_all[
                'fclass'
            ].astype(str)
        )
    )


# ------------------------------------------------------------
# Standardize X
# ------------------------------------------------------------

X_raw = (
    data_all[
        feature_cols
    ]
    .values
)

scaler_x = StandardScaler()

X_sc = scaler_x.fit_transform(
    X_raw
)


# ------------------------------------------------------------
# log1p + standardize y
# ------------------------------------------------------------

Y_raw = (
    data_all[
        'avg_flow'
    ]
    .values
)

Y_log = np.log1p(
    Y_raw
)

scaler_y = StandardScaler()

Y_sc = (
    scaler_y
    .fit_transform(
        Y_log.reshape(
            -1,
            1
        )
    )
    .flatten()
)


# ============================================================
# 8. 构建 road-segment graph
# ============================================================

print(
    ">>> 构建无向道路图..."
)

node_to_roads = {}

for idx, cid in enumerate(
    data_all['cid']
):

    for node in str(cid).split('_'):

        node_to_roads.setdefault(
            node,
            []
        ).append(
            idx
        )


edge_from = []
edge_to = []

for roads in node_to_roads.values():

    for r1 in roads:

        for r2 in roads:

            if r1 != r2:

                edge_from.append(
                    r1
                )

                edge_to.append(
                    r2
                )


edge_index_raw = torch.tensor(
    [
        edge_from,
        edge_to
    ],
    dtype=torch.long
)


edge_index = to_undirected(
    edge_index_raw
).to(device)


# ------------------------------------------------------------
# Node features tensor
# ------------------------------------------------------------

x_tensor = torch.tensor(
    X_sc,
    dtype=torch.float
).to(device)


# ------------------------------------------------------------
# Edge similarity:
# cosine similarity [-1,1] -> [0,1]
# ------------------------------------------------------------

edge_attr = (
    (
        F.cosine_similarity(
            x_tensor[
                edge_index[0]
            ],
            x_tensor[
                edge_index[1]
            ],
            dim=1
        )
        +
        1
    )
    /
    2
).view(
    -1,
    1
)


print(
    f">>> Number of nodes: "
    f"{X_sc.shape[0]}"
)

print(
    f">>> Number of edges: "
    f"{edge_index.shape[1]}"
)

print(
    f">>> Number of geographic features: "
    f"{X_sc.shape[1]}"
)


# ============================================================
# 9. Proposed GAT:
# Optuna hyperparameter optimization
#
# IMPORTANT:
# GAT training does NOT use traffic-flow labels.
# Therefore it is trained once per time-period dataset,
# rather than being retrained for every missing ratio.
# ============================================================

EMBEDDING_DIM = 32

GAT_OPTUNA_TRIALS = 15

GAT_TUNING_EPOCHS = 200

FINAL_GAT_EPOCHS = 400


print(
    "\n"
    +
    "=" * 80
)

print(
    ">>> Optimizing contrastive GAT encoder..."
)

print(
    "=" * 80
)


def gat_objective(
    trial
):

    hidden_dim = trial.suggest_int(
        'hid',
        32,
        128,
        step=32
    )

    num_layers = trial.suggest_int(
        'num_layers',
        2,
        4
    )

    heads = trial.suggest_categorical(
        'heads',
        [
            2,
            4,
            8
        ]
    )

    dropout = trial.suggest_float(
        'dropout',
        0.1,
        0.4
    )

    learning_rate = trial.suggest_float(
        'lr',
        1e-4,
        5e-3,
        log=True
    )

    temperature = trial.suggest_float(
        'temp',
        0.1,
        0.4
    )


    trial_seed = (
        SEED
        +
        trial.number
    )

    seed_everything(
        trial_seed
    )


    model = GATEncoder(
        in_c=X_sc.shape[1],
        hid_c=hidden_dim,
        out_c=EMBEDDING_DIM,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout
    ).to(device)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    model.train()


    for epoch in range(
        GAT_TUNING_EPOCHS
    ):

        optimizer.zero_grad()

        _, projection = model(
            x_tensor,
            edge_index,
            edge_attr
        )

        loss = contrastive_loss(
            projection=projection,
            edge_index=edge_index,
            temperature=temperature,
            num_nodes=X_sc.shape[0]
        )

        loss.backward()

        optimizer.step()


    final_loss = loss.item()


    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    return final_loss


sampler_gat = optuna.samplers.TPESampler(
    seed=SEED
)

study_gat = optuna.create_study(
    direction='minimize',
    sampler=sampler_gat
)

study_gat.optimize(
    gat_objective,
    n_trials=GAT_OPTUNA_TRIALS
)


best_gat = study_gat.best_params


print(
    "\n>>> Best GAT parameters:"
)

print(
    best_gat
)


# ============================================================
# 10. Re-train final GAT with best parameters
# ============================================================

print(
    "\n>>> Re-training final GAT encoder..."
)


seed_everything(
    SEED
)


final_gat = GATEncoder(
    in_c=X_sc.shape[1],
    hid_c=best_gat['hid'],
    out_c=EMBEDDING_DIM,
    num_layers=best_gat['num_layers'],
    heads=best_gat['heads'],
    dropout=best_gat['dropout']
).to(device)


optimizer_gat = torch.optim.Adam(
    final_gat.parameters(),
    lr=best_gat['lr']
)


final_gat.train()


for epoch in range(
    FINAL_GAT_EPOCHS + 1
):

    optimizer_gat.zero_grad()

    z, projection = final_gat(
        x_tensor,
        edge_index,
        edge_attr
    )

    loss_gat = contrastive_loss(
        projection=projection,
        edge_index=edge_index,
        temperature=best_gat['temp'],
        num_nodes=X_sc.shape[0]
    )

    loss_gat.backward()

    optimizer_gat.step()


    if epoch % 100 == 0:

        print(
            f"    Epoch {epoch:3d}/"
            f"{FINAL_GAT_EPOCHS} | "
            f"Contrastive loss = "
            f"{loss_gat.item():.6f}"
        )


# ------------------------------------------------------------
# Extract trained spatial embeddings
# ------------------------------------------------------------

final_gat.eval()


with torch.no_grad():

    spatial_emb = (
        final_gat(
            x_tensor,
            edge_index,
            edge_attr
        )[0]
        .detach()
        .cpu()
        .numpy()
    )


# ------------------------------------------------------------
# Proposed model input
# ------------------------------------------------------------

X_combined = np.hstack(
    [
        X_sc,
        spatial_emb
    ]
)


print(
    f">>> Spatial embedding shape: "
    f"{spatial_emb.shape}"
)

print(
    f">>> Combined feature shape: "
    f"{X_combined.shape}"
)


# ============================================================
# 11. Sparsity experiment settings
# ============================================================

missing_rates = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]


# 每个 missing ratio 独立重复5次
REPEATS = 5


# 每个 repeat 内使用5-fold ensemble
N_FOLDS = 5


# XGBoost Optuna
XGB_TRIALS = 10


# GAT-MLP Optuna
GAT_MLP_TRIALS = 10


results_collector = []


# ============================================================
# 12. Missing-ratio experiment
# ============================================================

for rate in missing_rates:

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        f">>> Missing ratio: "
        f"{rate * 100:.0f}%"
    )

    print(
        "=" * 80
    )


    for repeat in range(
        REPEATS
    ):

        print(
            f"\n--- Repeat "
            f"{repeat + 1}/{REPEATS} ---"
        )


        # ----------------------------------------------------
        # Reproducible random seed for this repeat
        # ----------------------------------------------------

        repeat_seed = (
            SEED
            +
            repeat
        )

        seed_everything(
            repeat_seed
        )


        indices = np.arange(
            X_sc.shape[0]
        )


        # ====================================================
        # A. 80% training pool + 20% fixed test
        # ====================================================

        pool_train, fixed_test = (
            train_test_split(
                indices,
                test_size=0.2,
                random_state=repeat_seed
            )
        )


        # ====================================================
        # B. Remove traffic-flow labels from training pool
        # ====================================================

        keep_num = max(
            int(
                len(pool_train)
                *
                (
                    1
                    -
                    rate
                )
            ),
            20
        )


        rng = np.random.default_rng(
            repeat_seed
            +
            int(
                rate
                *
                1000
            )
        )


        active_train = rng.choice(
            pool_train,
            size=keep_num,
            replace=False
        )


        print(
            f"    Training pool: "
            f"{len(pool_train)}"
        )

        print(
            f"    Active labeled roads: "
            f"{len(active_train)}"
        )

        print(
            f"    Fixed test roads: "
            f"{len(fixed_test)}"
        )


        # ====================================================
        # Five-fold CV within active labeled training roads
        # ====================================================

        kf = KFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=SEED
        )


        fold_predictions = {
            "GAT-XGBoost": [],
            "XGBoost": [],
            "GAT-MLP": []
        }


        # ====================================================
        # Fold loop
        # ====================================================

        for fold, (
            train_inner,
            val_inner
        ) in enumerate(
            kf.split(
                active_train
            ),
            start=1
        ):

            print(
                f"    Fold {fold}/{N_FOLDS}"
            )


            idx_train = active_train[
                train_inner
            ]

            idx_val = active_train[
                val_inner
            ]


            # =================================================
            # MODEL 1:
            # GAT-XGBoost
            # =================================================

            def xgb_gat_objective(
                trial
            ):

                params = {

                    'n_estimators':
                        trial.suggest_int(
                            'n_estimators',
                            500,
                            2000
                        ),

                    'max_depth':
                        trial.suggest_int(
                            'max_depth',
                            3,
                            12
                        ),

                    'learning_rate':
                        trial.suggest_float(
                            'learning_rate',
                            0.01,
                            0.2,
                            log=True
                        ),

                    'subsample':
                        trial.suggest_float(
                            'subsample',
                            0.5,
                            1.0
                        ),

                    'colsample_bytree':
                        trial.suggest_float(
                            'colsample_bytree',
                            0.5,
                            1.0
                        ),

                    'min_child_weight':
                        trial.suggest_int(
                            'min_child_weight',
                            1,
                            10
                        ),

                    'tree_method':
                        'hist',

                    'random_state':
                        SEED,

                    'n_jobs':
                        -1
                }


                model = xgb.XGBRegressor(
                    **params
                )


                model.fit(
                    X_combined[
                        idx_train
                    ],
                    Y_sc[
                        idx_train
                    ]
                )


                pred_val = model.predict(
                    X_combined[
                        idx_val
                    ]
                )


                return mean_squared_error(
                    Y_sc[
                        idx_val
                    ],
                    pred_val
                )


            sampler_x1 = (
                optuna.samplers.TPESampler(
                    seed=(
                        SEED
                        +
                        repeat * 100
                        +
                        fold
                    )
                )
            )


            study_x1 = (
                optuna.create_study(
                    direction='minimize',
                    sampler=sampler_x1
                )
            )


            study_x1.optimize(
                xgb_gat_objective,
                n_trials=XGB_TRIALS
            )


            best_x1 = (
                study_x1.best_params
                .copy()
            )


            best_x1.update(
                {
                    'tree_method':
                        'hist',

                    'random_state':
                        SEED,

                    'n_jobs':
                        -1
                }
            )


            model_gat_xgb = (
                xgb.XGBRegressor(
                    **best_x1
                )
            )


            model_gat_xgb.fit(
                X_combined[
                    idx_train
                ],
                Y_sc[
                    idx_train
                ]
            )


            pred_test_gat_xgb = (
                model_gat_xgb.predict(
                    X_combined[
                        fixed_test
                    ]
                )
            )


            fold_predictions[
                "GAT-XGBoost"
            ].append(
                pred_test_gat_xgb
            )


            # =================================================
            # MODEL 2:
            # Pure XGBoost
            # =================================================

            def xgb_pure_objective(
                trial
            ):

                params = {

                    'n_estimators':
                        trial.suggest_int(
                            'n_estimators',
                            500,
                            2000
                        ),

                    'max_depth':
                        trial.suggest_int(
                            'max_depth',
                            3,
                            12
                        ),

                    'learning_rate':
                        trial.suggest_float(
                            'learning_rate',
                            0.01,
                            0.2,
                            log=True
                        ),

                    'subsample':
                        trial.suggest_float(
                            'subsample',
                            0.5,
                            1.0
                        ),

                    'colsample_bytree':
                        trial.suggest_float(
                            'colsample_bytree',
                            0.5,
                            1.0
                        ),

                    'min_child_weight':
                        trial.suggest_int(
                            'min_child_weight',
                            1,
                            10
                        ),

                    'tree_method':
                        'hist',

                    'random_state':
                        SEED,

                    'n_jobs':
                        -1
                }


                model = (
                    xgb.XGBRegressor(
                        **params
                    )
                )


                model.fit(
                    X_sc[
                        idx_train
                    ],
                    Y_sc[
                        idx_train
                    ]
                )


                pred_val = model.predict(
                    X_sc[
                        idx_val
                    ]
                )


                return mean_squared_error(
                    Y_sc[
                        idx_val
                    ],
                    pred_val
                )


            sampler_x2 = (
                optuna.samplers.TPESampler(
                    seed=(
                        SEED
                        +
                        10000
                        +
                        repeat * 100
                        +
                        fold
                    )
                )
            )


            study_x2 = (
                optuna.create_study(
                    direction='minimize',
                    sampler=sampler_x2
                )
            )


            study_x2.optimize(
                xgb_pure_objective,
                n_trials=XGB_TRIALS
            )


            best_x2 = (
                study_x2.best_params
                .copy()
            )


            best_x2.update(
                {
                    'tree_method':
                        'hist',

                    'random_state':
                        SEED,

                    'n_jobs':
                        -1
                }
            )


            model_xgb = (
                xgb.XGBRegressor(
                    **best_x2
                )
            )


            model_xgb.fit(
                X_sc[
                    idx_train
                ],
                Y_sc[
                    idx_train
                ]
            )


            pred_test_xgb = (
                model_xgb.predict(
                    X_sc[
                        fixed_test
                    ]
                )
            )


            fold_predictions[
                "XGBoost"
            ].append(
                pred_test_xgb
            )


            # =================================================
            # MODEL 3:
            # GAT-MLP
            #
            # End-to-end supervised model
            # Huber loss is used consistently with Table 2
            # =================================================

            def gat_mlp_objective(
                trial
            ):

                hidden_channels = (
                    trial.suggest_int(
                        'hidden_channels',
                        32,
                        128,
                        step=16
                    )
                )

                heads = (
                    trial.suggest_categorical(
                        'heads',
                        [
                            2,
                            4,
                            8
                        ]
                    )
                )

                learning_rate = (
                    trial.suggest_float(
                        'lr',
                        1e-4,
                        5e-3,
                        log=True
                    )
                )

                dropout = (
                    trial.suggest_float(
                        'dropout',
                        0.1,
                        0.4
                    )
                )

                num_layers = (
                    trial.suggest_int(
                        'num_layers',
                        2,
                        4
                    )
                )


                trial_seed = (
                    SEED
                    +
                    20000
                    +
                    repeat * 100
                    +
                    fold * 10
                    +
                    trial.number
                )


                seed_everything(
                    trial_seed
                )


                model = GATMLPModel(
                    in_c=X_sc.shape[1],
                    hidden_channels=hidden_channels,
                    embedding_dim=EMBEDDING_DIM,
                    num_layers=num_layers,
                    heads=heads,
                    dropout=dropout
                ).to(device)


                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=1e-3
                )


                criterion = nn.HuberLoss()


                model.train()


                train_tensor = torch.tensor(
                    idx_train,
                    dtype=torch.long,
                    device=device
                )


                for epoch in range(
                    150
                ):

                    optimizer.zero_grad()

                    output = model(
                        x_tensor,
                        edge_index,
                        edge_attr
                    ).flatten()


                    loss = criterion(
                        output[
                            train_tensor
                        ],
                        torch.tensor(
                            Y_sc[
                                idx_train
                            ],
                            dtype=torch.float,
                            device=device
                        )
                    )


                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0
                    )

                    optimizer.step()


                model.eval()


                val_tensor = torch.tensor(
                    idx_val,
                    dtype=torch.long,
                    device=device
                )


                with torch.no_grad():

                    pred_val = (
                        model(
                            x_tensor,
                            edge_index,
                            edge_attr
                        )
                        .flatten()[
                            val_tensor
                        ]
                        .cpu()
                        .numpy()
                    )


                val_mae = mean_absolute_error(
                    Y_sc[
                        idx_val
                    ],
                    pred_val
                )


                del model

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


                return val_mae


            sampler_mlp = (
                optuna.samplers.TPESampler(
                    seed=(
                        SEED
                        +
                        20000
                        +
                        repeat * 100
                        +
                        fold
                    )
                )
            )


            study_mlp = (
                optuna.create_study(
                    direction='minimize',
                    sampler=sampler_mlp
                )
            )


            study_mlp.optimize(
                gat_mlp_objective,
                n_trials=GAT_MLP_TRIALS
            )


            best_mlp = (
                study_mlp.best_params
            )


            # -------------------------------------------------
            # Final GAT-MLP
            # -------------------------------------------------

            seed_everything(
                SEED
                +
                30000
                +
                repeat * 100
                +
                fold
            )


            model_gat_mlp = (
                GATMLPModel(
                    in_c=X_sc.shape[1],

                    hidden_channels=
                        best_mlp[
                            'hidden_channels'
                        ],

                    embedding_dim=
                        EMBEDDING_DIM,

                    num_layers=
                        best_mlp[
                            'num_layers'
                        ],

                    heads=
                        best_mlp[
                            'heads'
                        ],

                    dropout=
                        best_mlp[
                            'dropout'
                        ]
                )
                .to(device)
            )


            optimizer_mlp = (
                torch.optim.Adam(
                    model_gat_mlp.parameters(),
                    lr=best_mlp[
                        'lr'
                    ],
                    weight_decay=1e-3
                )
            )


            criterion = nn.HuberLoss()


            train_tensor = torch.tensor(
                idx_train,
                dtype=torch.long,
                device=device
            )


            target_train = torch.tensor(
                Y_sc[
                    idx_train
                ],
                dtype=torch.float,
                device=device
            )


            model_gat_mlp.train()


            for epoch in range(
                300
            ):

                optimizer_mlp.zero_grad()

                output = (
                    model_gat_mlp(
                        x_tensor,
                        edge_index,
                        edge_attr
                    )
                    .flatten()
                )


                loss = criterion(
                    output[
                        train_tensor
                    ],
                    target_train
                )


                loss.backward()


                torch.nn.utils.clip_grad_norm_(
                    model_gat_mlp.parameters(),
                    max_norm=1.0
                )


                optimizer_mlp.step()


            # -------------------------------------------------
            # Fixed-test prediction
            # -------------------------------------------------

            test_tensor = torch.tensor(
                fixed_test,
                dtype=torch.long,
                device=device
            )


            model_gat_mlp.eval()


            with torch.no_grad():

                pred_test_gat_mlp = (
                    model_gat_mlp(
                        x_tensor,
                        edge_index,
                        edge_attr
                    )
                    .flatten()[
                        test_tensor
                    ]
                    .cpu()
                    .numpy()
                )


            fold_predictions[
                "GAT-MLP"
            ].append(
                pred_test_gat_mlp
            )


            # -------------------------------------------------
            # Cleanup
            # -------------------------------------------------

            del model_gat_xgb
            del model_xgb
            del model_gat_mlp

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        # ====================================================
        # Average predictions across 5 folds
        # ====================================================

        for model_name in [
            "GAT-XGBoost",
            "XGBoost",
            "GAT-MLP"
        ]:

            avg_prediction = np.mean(
                np.asarray(
                    fold_predictions[
                        model_name
                    ]
                ),
                axis=0
            )


            metrics = compute_all_metrics(
                y_true_sc=Y_sc[
                    fixed_test
                ],
                y_pred_sc=avg_prediction,
                scaler_y=scaler_y
            )


            for metric_name, metric_value in (
                metrics.items()
            ):

                results_collector.append(
                    {
                        "Missing_Rate":
                            rate,

                        "Repeat":
                            repeat + 1,

                        "Model":
                            model_name,

                        "Metric":
                            metric_name,

                        "Value":
                            metric_value
                    }
                )


            print(
                f"    {model_name}: "
                f"RMSE={metrics['RMSE']:.2f}, "
                f"MAE={metrics['MAE']:.2f}, "
                f"MAPE={metrics['MAPE']:.2f}%"
            )


# ============================================================
# 13. 保存每次 repeat 的原始结果
# ============================================================

df_results = pd.DataFrame(
    results_collector
)


raw_save_path = os.path.join(
    save_dir,
    'sparsity_all_repeats_corrected.csv'
)


df_results.to_csv(
    raw_save_path,
    index=False
)


# ============================================================
# 14. Mean ± SD across five repetitions
# ============================================================

final_summary = (
    df_results
    .groupby(
        [
            "Missing_Rate",
            "Model",
            "Metric"
        ]
    )[
        "Value"
    ]
    .agg(
        [
            "mean",
            "std"
        ]
    )
    .reset_index()
)


# ------------------------------------------------------------
# Wide-format output:
#
# Missing_Rate | Model |
# RMSE_mean | RMSE_std |
# MAE_mean  | MAE_std  |
# MAPE_mean | MAPE_std |
# R2_mean   | R2_std
# ------------------------------------------------------------

final_pivot = (
    final_summary
    .pivot_table(
        index=[
            "Missing_Rate",
            "Model"
        ],

        columns="Metric",

        values=[
            "mean",
            "std"
        ]
    )
)


final_pivot.columns = [
    f"{metric}_{stat}"
    for stat, metric
    in final_pivot.columns
]


final_pivot = (
    final_pivot
    .reset_index()
)


summary_save_path = os.path.join(
    save_dir,
    'sparsity_nested_optimized_results_corrected.csv'
)


final_pivot.to_csv(
    summary_save_path,
    index=False
)


# ============================================================
# 15. 保存 GAT 最优参数
# ============================================================

gat_params_save = os.path.join(
    save_dir,
    'sparsity_gat_best_params.csv'
)


pd.DataFrame(
    [
        best_gat
    ]
).to_csv(
    gat_params_save,
    index=False
)


# ============================================================
# 16. 输出结果
# ============================================================

print(
    "\n"
    +
    "=" * 80
)

print(
    "✅ Sparsity experiment completed"
)

print(
    f"✅ Raw repeat results:\n"
    f"{raw_save_path}"
)

print(
    f"✅ Mean ± SD summary:\n"
    f"{summary_save_path}"
)

print(
    f"✅ GAT best parameters:\n"
    f"{gat_params_save}"
)

print(
    "=" * 80
)


print(
    "\nFinal summary:"
)

print(
    final_pivot.to_string(
        index=False
    )
)

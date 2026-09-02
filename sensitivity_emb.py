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
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import KFold


# ============================================================
# 1. 环境配置
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
# 2. 路径设置
# ============================================================

road_path = (
    r'D:\KEKE\宣城交通流量预测\data\road_final_features.csv'
)

traffic_path = (
    r'D:\KEKE\宣城交通流量预测\data\Evening_Average.csv'
)

root_save_dir = (
    rf"results\interp\evening\seed_{SEED}\sensitivity_analysis"
)

os.makedirs(
    root_save_dir,
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
        dropout=0.2
    ):
        super().__init__()

        self.num_layers = num_layers

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
        # 最后一层
        # heads = 1
        # 输出维度 = embedding dimension
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
        # Projection head
        # 只用于 contrastive learning
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

        # ----------------------------------------------------
        # Hidden GAT layers
        # ----------------------------------------------------

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
        # Spatial embedding
        # ----------------------------------------------------

        z = self.convs[-1](
            x,
            edge_index,
            edge_attr
        )

        # ----------------------------------------------------
        # Projection for InfoNCE
        # ----------------------------------------------------

        p = self.proj(z)

        return z, p


# ============================================================
# 4. 指标计算
# ============================================================

def compute_all_metrics(
    y_true_sc,
    y_pred_sc,
    scaler_y
):

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
# 5. InfoNCE-based contrastive loss
# ============================================================

def contrastive_loss(
    projection,
    edge_index,
    temperature,
    num_nodes
):

    # --------------------------------------------------------
    # L2 normalize projection
    # --------------------------------------------------------

    p_norm = F.normalize(
        projection,
        p=2,
        dim=1
    )

    # --------------------------------------------------------
    # Positive:
    # topologically connected roads
    # --------------------------------------------------------

    pos_score = torch.sum(
        p_norm[edge_index[0]]
        *
        p_norm[edge_index[1]],
        dim=-1
    )

    pos_score = (
        pos_score
        /
        temperature
    )

    # --------------------------------------------------------
    # Negative:
    # randomly sampled non-connected road pairs
    # --------------------------------------------------------

    neg_edge = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=edge_index.size(1)
    )

    neg_score = torch.sum(
        p_norm[neg_edge[0]]
        *
        p_norm[neg_edge[1]],
        dim=-1
    )

    neg_score = (
        neg_score
        /
        temperature
    )

    # --------------------------------------------------------
    # 数值稳定形式
    #
    # 等价于：
    #
    # -log[
    # exp(pos) /
    # (exp(pos)+exp(neg))
    # ]
    # --------------------------------------------------------

    loss = F.softplus(
        neg_score
        -
        pos_score
    ).mean()

    return loss


# ============================================================
# 6. 最终 GAT retraining
# ============================================================

def train_final_gat(
    x_tensor,
    edge_index,
    edge_attr,

    hidden_channels,
    embedding_dim,
    num_layers,
    heads,

    learning_rate,
    temperature,

    dropout=0.2,
    epochs=400,
    seed=42
):

    seed_everything(
        seed
    )

    model = GATEncoder(
        in_c=x_tensor.shape[1],
        hid_c=hidden_channels,
        out_c=embedding_dim,
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
        epochs + 1
    ):

        optimizer.zero_grad()

        z, projection = model(
            x_tensor,
            edge_index,
            edge_attr
        )

        loss = contrastive_loss(
            projection=projection,
            edge_index=edge_index,
            temperature=temperature,
            num_nodes=x_tensor.shape[0]
        )

        loss.backward()

        optimizer.step()


        if epoch % 100 == 0:

            print(
                f"        Epoch "
                f"{epoch:3d}/{epochs} | "
                f"Loss={loss.item():.6f}"
            )


    # --------------------------------------------------------
    # 正式提取 embedding 时关闭 dropout
    # --------------------------------------------------------

    model.eval()


    with torch.no_grad():

        embedding, _ = model(
            x_tensor,
            edge_index,
            edge_attr
        )


    embedding = (
        embedding
        .detach()
        .cpu()
        .numpy()
    )


    return (
        model,
        embedding,
        loss.item()
    )


# ============================================================
# 7. 数据准备
# ============================================================

print(
    ">>> 正在初始化数据..."
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
# Road class encoding
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
# log1p + standardize traffic flow
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
    ">>> 正在构建道路图..."
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
# Node features
# ------------------------------------------------------------

x_tensor = torch.tensor(
    X_sc,
    dtype=torch.float
).to(device)


# ------------------------------------------------------------
# Edge geographic similarity:
# cosine [-1,1] -> [0,1]
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
    f">>> Number of road segments: "
    f"{X_sc.shape[0]}"
)

print(
    f">>> Number of edges: "
    f"{edge_index.shape[1]}"
)

print(
    f">>> Number of features: "
    f"{X_sc.shape[1]}"
)


# ============================================================
# 9. Embedding-dimension sensitivity settings
# ============================================================

# 与 Figure 4 一致
embedding_dims = [
    8,
    16,
    32,
    64,
    128
]


# ------------------------------------------------------------
# 层数固定
# ------------------------------------------------------------

FIXED_LAYERS = 3


# ------------------------------------------------------------
# Dropout 固定
# ------------------------------------------------------------

DROPOUT = 0.2


# ------------------------------------------------------------
# GAT optimization settings
# ------------------------------------------------------------

GAT_OPTUNA_TRIALS = 10

GAT_TUNING_EPOCHS = 150

FINAL_GAT_EPOCHS = 400


# ------------------------------------------------------------
# XGBoost optimization settings
# ------------------------------------------------------------

XGB_OPTUNA_TRIALS = 8


sensitivity_results = []


# ============================================================
# 10. Embedding dimension sensitivity loop
# ============================================================

for embedding_dim in embedding_dims:

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        f">>> Testing Embedding Dimension = "
        f"{embedding_dim}"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # 10.1 Optimize remaining GAT hyperparameters
    # ========================================================

    def gat_objective(
        trial
    ):

        hidden_channels = (
            trial.suggest_categorical(
                'hidden_channels',
                [
                    8,
                    16,
                    32,
                    64,
                    128,
                    256
                ]
            )
        )


        heads = (
            trial.suggest_categorical(
                'heads',
                [
                    1,
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


        temperature = (
            trial.suggest_float(
                'temp',
                0.1,
                0.4
            )
        )


        trial_seed = (
            SEED
            +
            embedding_dim * 100
            +
            trial.number
        )


        seed_everything(
            trial_seed
        )


        model = GATEncoder(
            in_c=X_sc.shape[1],
            hid_c=hidden_channels,
            out_c=embedding_dim,
            num_layers=FIXED_LAYERS,
            heads=heads,
            dropout=DROPOUT
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


    # --------------------------------------------------------
    # Optuna GAT
    # --------------------------------------------------------

    sampler_gat = (
        optuna.samplers.TPESampler(
            seed=(
                SEED
                +
                embedding_dim
            )
        )
    )


    study_gat = (
        optuna.create_study(
            direction='minimize',
            sampler=sampler_gat
        )
    )


    study_gat.optimize(
        gat_objective,
        n_trials=GAT_OPTUNA_TRIALS
    )


    best_gat = (
        study_gat.best_params
    )


    print(
        f">>> Best GAT parameters: "
        f"{best_gat}"
    )


    # ========================================================
    # 10.2 IMPORTANT:
    # Re-train final GAT
    # ========================================================

    print(
        ">>> Re-training final GAT "
        "with best parameters..."
    )


    final_gat, embedding, final_gat_loss = (
        train_final_gat(

            x_tensor=x_tensor,

            edge_index=edge_index,

            edge_attr=edge_attr,

            hidden_channels=
                best_gat[
                    'hidden_channels'
                ],

            embedding_dim=
                embedding_dim,

            num_layers=
                FIXED_LAYERS,

            heads=
                best_gat[
                    'heads'
                ],

            learning_rate=
                best_gat[
                    'lr'
                ],

            temperature=
                best_gat[
                    'temp'
                ],

            dropout=
                DROPOUT,

            epochs=
                FINAL_GAT_EPOCHS,

            seed=(
                SEED
                +
                embedding_dim * 1000
            )
        )
    )


    print(
        f">>> Final GAT loss: "
        f"{final_gat_loss:.6f}"
    )


    print(
        f">>> Embedding shape: "
        f"{embedding.shape}"
    )


    # --------------------------------------------------------
    # Geographic features + trained embedding
    # --------------------------------------------------------

    X_combined = np.hstack(
        [
            X_sc,
            embedding
        ]
    )


    # ========================================================
    # 10.3 5-fold CV
    # ========================================================

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )


    fold_metrics = []


    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        kf.split(
            X_combined
        ),
        start=1
    ):

        print(
            f">>> Embedding={embedding_dim} | "
            f"Fold {fold}/5"
        )


        x_train = X_combined[
            train_idx
        ]

        x_val = X_combined[
            val_idx
        ]

        y_train = Y_sc[
            train_idx
        ]

        y_val = Y_sc[
            val_idx
        ]


        # ====================================================
        # XGBoost tuning
        # ====================================================

        def xgb_objective(
            trial
        ):

            params = {

                'n_estimators':
                    trial.suggest_int(
                        'n_estimators',
                        500,
                        1500
                    ),

                'max_depth':
                    trial.suggest_int(
                        'max_depth',
                        3,
                        10
                    ),

                'learning_rate':
                    trial.suggest_float(
                        'learning_rate',
                        0.01,
                        0.1,
                        log=True
                    ),

                'subsample':
                    trial.suggest_float(
                        'subsample',
                        0.6,
                        1.0
                    ),

                'colsample_bytree':
                    trial.suggest_float(
                        'colsample_bytree',
                        0.6,
                        1.0
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
                x_train,
                y_train
            )


            prediction = model.predict(
                x_val
            )


            return mean_squared_error(
                y_val,
                prediction
            )


        sampler_xgb = (
            optuna.samplers.TPESampler(
                seed=(
                    SEED
                    +
                    embedding_dim * 100
                    +
                    fold
                )
            )
        )


        study_xgb = (
            optuna.create_study(
                direction='minimize',
                sampler=sampler_xgb
            )
        )


        study_xgb.optimize(
            xgb_objective,
            n_trials=XGB_OPTUNA_TRIALS
        )


        # ----------------------------------------------------
        # Final XGBoost for this fold
        # ----------------------------------------------------

        best_xgb = (
            study_xgb.best_params
            .copy()
        )


        best_xgb.update(
            {
                'tree_method':
                    'hist',

                'random_state':
                    SEED,

                'n_jobs':
                    -1
            }
        )


        model_xgb = xgb.XGBRegressor(
            **best_xgb
        )


        model_xgb.fit(
            x_train,
            y_train
        )


        prediction = model_xgb.predict(
            x_val
        )


        result = compute_all_metrics(
            y_true_sc=y_val,
            y_pred_sc=prediction,
            scaler_y=scaler_y
        )


        result[
            'Fold'
        ] = fold


        fold_metrics.append(
            result
        )


        print(
            f"    RMSE="
            f"{result['RMSE']:.2f} | "
            f"MAE="
            f"{result['MAE']:.2f} | "
            f"MAPE="
            f"{result['MAPE']:.2f}% | "
            f"R2="
            f"{result['R2']:.4f}"
        )


        del model_xgb

        gc.collect()


    # ========================================================
    # 10.4 Five-fold mean ± SD
    # ========================================================

    fold_df = pd.DataFrame(
        fold_metrics
    )


    metric_names = [
        'RMSE',
        'MAE',
        'MAPE',
        'R2'
    ]


    mean_metrics = (
        fold_df[
            metric_names
        ]
        .mean()
        .to_dict()
    )


    std_metrics = (
        fold_df[
            metric_names
        ]
        .std()
        .to_dict()
    )


    result_row = {

        "Embedding_Dim":
            embedding_dim,

        "Best_hidden_channels":
            best_gat[
                'hidden_channels'
            ],

        "Best_heads":
            best_gat[
                'heads'
            ],

        "Best_learning_rate":
            best_gat[
                'lr'
            ],

        "Best_temperature":
            best_gat[
                'temp'
            ],

        "Final_GAT_loss":
            final_gat_loss
    }


    for metric in metric_names:

        result_row[
            f"{metric}_mean"
        ] = mean_metrics[
            metric
        ]

        result_row[
            f"{metric}_std"
        ] = std_metrics[
            metric
        ]


    sensitivity_results.append(
        result_row
    )


    # --------------------------------------------------------
    # 保存该 embedding dimension 的5折结果
    # --------------------------------------------------------

    fold_save_path = os.path.join(
        root_save_dir,
        (
            f"embedding_{embedding_dim}"
            f"_fold_metrics.csv"
        )
    )


    fold_df.to_csv(
        fold_save_path,
        index=False
    )


    print(
        "\n"
        f">>> Embedding dimension "
        f"{embedding_dim} completed"
    )


    print(
        f"    RMSE = "
        f"{mean_metrics['RMSE']:.2f} "
        f"± "
        f"{std_metrics['RMSE']:.2f}"
    )


    print(
        f"    MAE  = "
        f"{mean_metrics['MAE']:.2f} "
        f"± "
        f"{std_metrics['MAE']:.2f}"
    )


    print(
        f"    MAPE = "
        f"{mean_metrics['MAPE']:.2f}% "
        f"± "
        f"{std_metrics['MAPE']:.2f}%"
    )


    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del final_gat
    del embedding
    del X_combined

    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# 11. 保存最终 sensitivity results
# ============================================================

final_df = pd.DataFrame(
    sensitivity_results
)


save_path = os.path.join(
    root_save_dir,
    "embedding_sensitivity_results_corrected.csv"
)


final_df.to_csv(
    save_path,
    index=False
)


print(
    "\n"
    +
    "=" * 70
)

print(
    "✅ Embedding dimension sensitivity analysis completed"
)

print(
    f"✅ Saved to:\n{save_path}"
)

print(
    "=" * 70
)


# ============================================================
# 12. 显示最终结果
# ============================================================

display_columns = [
    "Embedding_Dim",

    "RMSE_mean",
    "RMSE_std",

    "MAE_mean",
    "MAE_std",

    "MAPE_mean",
    "MAPE_std",

    "R2_mean",
    "R2_std"
]


print(
    "\nFinal results:"
)


print(
    final_df[
        display_columns
    ].to_string(
        index=False
    )
)
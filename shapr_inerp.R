# ============================================================
# R 脚本: 使用 shapr 计算 Group Shapley 值
# 保存为: run_shapr.R
# ============================================================

# ============================================================
# 1. 设置路径
# ============================================================
data_dir <- "D:/KEKE/宣城交通流量预测/results/interp/noon/seed_42/for_r/r_shapr_data"

if (!dir.exists(data_dir)) {
  stop(paste("错误：数据目录不存在:", data_dir))
}

output_dir <- file.path(data_dir, "results")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("====================================\n")
cat("开始运行 shapr Group Shapley 分析\n")
cat("====================================\n\n")
cat(sprintf("数据目录: %s\n", data_dir))
cat(sprintf("输出目录: %s\n\n", output_dir))

# ============================================================
# 2. 读取数据
# ============================================================
cat("读取数据...\n")

X_train_dt <- data.table::fread(file.path(data_dir, "X_train.csv"))
X_test_dt <- data.table::fread(file.path(data_dir, "X_test.csv"))
y_train <- data.table::fread(file.path(data_dir, "y_train.csv"))$target

cat(sprintf("  X_train: %d 行 x %d 列\n", nrow(X_train_dt), ncol(X_train_dt)))
cat(sprintf("  X_test: %d 行 x %d 列\n", nrow(X_test_dt), ncol(X_test_dt)))
cat(sprintf("  y_train: %d 个样本\n", length(y_train)))

# ============================================================
# 3. 读取 Group 信息
# ============================================================
cat("\n读取 Group 信息...\n")

group_mapping <- data.table::fread(file.path(data_dir, "group_mapping.csv"))
group_info <- jsonlite::fromJSON(file.path(data_dir, "group_info.json"))

unique_groups <- unique(group_mapping$group)
cat(sprintf("  总组数: %d\n", length(unique_groups)))

group_list <- list()
for (g in unique_groups) {
  features_in_group <- group_mapping$feature[group_mapping$group == g]
  group_list[[g]] <- as.character(features_in_group)
  
  if (g == "graph_embedding") {
    cat(sprintf("    %s: %d 个特征 (embedding)\n", g, length(features_in_group)))
  } else {
    cat(sprintf("    %s: %d 个特征\n", g, length(features_in_group)))
  }
}

phi0 <- group_info$phi0
cat(sprintf("\n  phi0 (baseline): %.4f\n", phi0))

# ============================================================
# 4. 读取 XGBoost 参数并训练模型
# ============================================================
cat("\n读取 XGBoost 参数...\n")

xgb_params_file <- file.path(data_dir, "xgb_params.json")
if (file.exists(xgb_params_file)) {
  xgb_params_python <- jsonlite::fromJSON(xgb_params_file)
  cat("  从 Python 读取的参数:\n")
  for (name in names(xgb_params_python)) {
    cat(sprintf("    %s: %s\n", name, xgb_params_python[[name]]))
  }
} else {
  cat("  ⚠️ 未找到参数文件，使用默认参数\n")
  xgb_params_python <- list(
    n_estimators = 100,
    max_depth = 6,
    learning_rate = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 42
  )
}

cat("\n训练 XGBoost 模型...\n")

# 使用 xgboost() 而不是 xgb.train()，这样 shapr 能更好地识别
model <- xgboost::xgboost(
  data = as.matrix(X_train_dt),
  label = y_train,
  max_depth = xgb_params_python$max_depth,
  eta = xgb_params_python$learning_rate,
  subsample = xgb_params_python$subsample,
  colsample_bytree = xgb_params_python$colsample_bytree,
  nrounds = xgb_params_python$n_estimators,
  objective = "reg:squarederror",
  verbose = 0
)

test_pred <- predict(model, as.matrix(X_test_dt[1:5, ]))
cat(sprintf("  R 模型预测 (前5个): %s\n", paste(round(test_pred, 2), collapse = ", ")))

# ============================================================
# 5. 准备解释数据
# ============================================================
cat("\n准备解释数据...\n")

max_explain <- 500
if (nrow(X_test_dt) > max_explain) {
  cat(sprintf("  测试集较大 (%d)，采样 %d 个样本进行解释\n", nrow(X_test_dt), max_explain))
  set.seed(42)
  sample_idx <- sample(1:nrow(X_test_dt), max_explain)
  x_explain <- X_test_dt[sample_idx, ]
} else {
  x_explain <- X_test_dt
  sample_idx <- 1:nrow(X_test_dt)
}

cat(sprintf("  解释样本数: %d\n", nrow(x_explain)))

# ============================================================
# 6. 定义自定义预测函数（新版 shapr 需要）
# ============================================================
# 为 xgb.Booster 定义 predict_model 方法
predict_model.xgb.Booster <- function(x, newdata) {
  if (is.data.frame(newdata) || data.table::is.data.table(newdata)) {
    newdata <- as.matrix(newdata)
  }
  predict(x, newdata)
}

# 为 xgb.Booster 定义 get_model_specs 方法
get_model_specs.xgb.Booster <- function(x) {
  list(
    labels = NA,
    classes = NA,
    factor_levels = NA
  )
}

# ============================================================
# 7. 计算 Group Shapley 值
# ============================================================
cat("\n计算 Group Shapley 值（这可能需要几分钟）...\n")

start_time <- Sys.time()

# 尝试使用新版 API
tryCatch({
  explanation <- shapr::explain(
    model = model,
    x_explain = x_explain,
    x_train = X_train_dt,
    approach = "empirical",
    phi0 = phi0,
    group = group_list,
    predict_model = function(model, newdata) {
      if (is.data.frame(newdata) || data.table::is.data.table(newdata)) {
        newdata <- as.matrix(newdata)
      }
      predict(model, newdata)
    },
    get_model_specs = function(x) {
      list(labels = NA, classes = NA, factor_levels = NA)
    },
    seed = 42
  )
}, error = function(e) {
  cat(sprintf("  第一种方法失败: %s\n", e$message))
  cat("  尝试备用方法...\n")
  
  # 备用方法：不使用 group，直接计算所有特征
  explanation <<- shapr::explain(
    model = model,
    x_explain = x_explain,
    x_train = X_train_dt,
    approach = "empirical",
    phi0 = phi0,
    predict_model = function(model, newdata) {
      if (is.data.frame(newdata) || data.table::is.data.table(newdata)) {
        newdata <- as.matrix(newdata)
      }
      predict(model, newdata)
    },
    get_model_specs = function(x) {
      list(labels = NA, classes = NA, factor_levels = NA)
    },
    seed = 42
  )
})

end_time <- Sys.time()
cat(sprintf("  ✅ 计算完成! 耗时: %.2f 分钟\n", 
            as.numeric(difftime(end_time, start_time, units = "mins"))))

# ============================================================
# 8. 提取并保存结果
# ============================================================
cat("\n保存结果...\n")

# 获取 Shapley 值
if (!is.null(explanation$shapley_values)) {
  shap_dt <- data.table::as.data.table(explanation$shapley_values)
} else if (!is.null(explanation$dt)) {
  shap_dt <- explanation$dt
} else {
  shap_dt <- data.table::as.data.table(explanation)
}

# 获取所有列名
all_cols <- colnames(shap_dt)
cat("  原始列名:\n")
print(all_cols)

# 提取 Shapley 值列（shapley_values_est.* 开头，但排除 explain_id 和 none）
est_cols <- grep("^shapley_values_est\\.", all_cols, value = TRUE)
est_cols <- est_cols[!grepl("explain_id|none", est_cols)]

# 提取标准差列
sd_cols <- grep("^shapley_values_sd\\.", all_cols, value = TRUE)
sd_cols <- sd_cols[!grepl("explain_id|none", sd_cols)]

cat(sprintf("\n  Shapley 值列数: %d\n", length(est_cols)))
cat(sprintf("  标准差列数: %d\n", length(sd_cols)))

# 创建干净的 Shapley 值 DataFrame
shap_values_df <- shap_dt[, ..est_cols]

# 重命名列（去掉前缀 "shapley_values_est."）
clean_names <- gsub("^shapley_values_est\\.", "", est_cols)
colnames(shap_values_df) <- clean_names

cat("\n  清理后的列名:\n")
print(clean_names)

# 保存完整的 Shapley 值
data.table::fwrite(shap_values_df, file.path(output_dir, "shap_values_grouped.csv"))
cat(sprintf("\n  ✅ Shapley 值保存至: %s\n", file.path(output_dir, "shap_values_grouped.csv")))

# 同时保存标准差
shap_sd_df <- shap_dt[, ..sd_cols]
colnames(shap_sd_df) <- gsub("^shapley_values_sd\\.", "", sd_cols)
data.table::fwrite(shap_sd_df, file.path(output_dir, "shap_values_sd.csv"))
cat(sprintf("  ✅ Shapley 值标准差保存至: %s\n", file.path(output_dir, "shap_values_sd.csv")))

# ============================================================
# 9. 计算特征重要性
# ============================================================
cat("\n计算特征重要性...\n")

shap_matrix <- as.matrix(shap_values_df)
sd_matrix <- as.matrix(shap_sd_df)

importance_df <- data.frame(
  feature_group = clean_names,
  importance_mean = colMeans(abs(shap_matrix), na.rm = TRUE),
  importance_std = apply(shap_matrix, 2, sd, na.rm = TRUE),
  shapley_sd_mean = colMeans(sd_matrix, na.rm = TRUE),  # 平均估计标准差
  num_features = sapply(clean_names, function(g) {
    if (g %in% names(group_list)) {
      length(group_list[[g]])
    } else {
      1
    }
  })
)

# 按重要性排序
importance_df <- importance_df[order(-importance_df$importance_mean), ]
rownames(importance_df) <- NULL
importance_df$rank <- 1:nrow(importance_df)

# 保存
data.table::fwrite(importance_df, file.path(output_dir, "feature_importance_grouped.csv"))
cat(sprintf("  ✅ 特征重要性保存至: %s\n", file.path(output_dir, "feature_importance_grouped.csv")))

cat("\n  特征重要性排名:\n")
print(importance_df)

# ============================================================
# 10. 保存非线性关系数据 (Dependence Data) - 增强匹配版
# ============================================================
cat("\n保存非线性关系数据...\n")

nonlinear_dir <- file.path(output_dir, "nonlinear_data")
dir.create(nonlinear_dir, showWarnings = FALSE, recursive = TRUE)

# 获取解释样本（x_explain）中实际存在的列名
actual_cols <- colnames(x_explain)

saved_count <- 0

# 直接遍历 shapr 算出的组名 (clean_names)
for (feat in clean_names) {
  
  # 1. 排除多维的 embedding 组
  if (feat == "graph_embedding") next
  
  # 2. 检查这个组名是否存在于我们的特征数据列中
  # 使用 match 确保精确匹配，包括中文处理
  if (feat %in% actual_cols) {
    
    # 提取特征值和 SHAP 值
    feat_values <- x_explain[[feat]]
    shap_values <- shap_values_df[[feat]]
    
    # 获取该特征在重要性表中的排名，用于文件名前缀
    rank_row <- which(importance_df$feature_group == feat)
    rank_val <- if (length(rank_row) > 0) importance_df$rank[rank_row] else 99
    
    # 构造依赖图数据框
    dep_df <- data.frame(
      feature_value = feat_values,
      shap_value = shap_values
    )
    
    # 确保文件名合法
    safe_feat_name <- gsub("[^[:alnum:]_\\u4e00-\\u9fa5]", "_", feat)
    filename <- sprintf("%02d_%s_dependence.csv", rank_val, safe_feat_name)
    
    # 保存数据
    data.table::fwrite(dep_df, file.path(nonlinear_dir, filename))
    saved_count <- saved_count + 1
    
  } else {
    # 如果没匹配上，打印出来调试，看看是哪个名字对不上
    cat(sprintf("  ⚠️ 跳过特征 [%s]: 在测试集列名中未找到精确匹配。\n", feat))
  }
}

cat(sprintf("  ✅ 成功保存了 %d 个特征的非线性数据至: %s\n", saved_count, nonlinear_dir))

# ============================================================
# 11. 保存采样索引和预测值
# ============================================================
# 保存采样索引
data.table::fwrite(data.frame(sample_idx = sample_idx), file.path(output_dir, "sample_indices.csv"))

# 保存预测值（如果有）
if ("pred_explain" %in% colnames(shap_dt)) {
  pred_df <- data.frame(
    sample_idx = sample_idx,
    prediction = shap_dt$pred_explain
  )
  data.table::fwrite(pred_df, file.path(output_dir, "predictions.csv"))
  cat("  ✅ 预测值已保存\n")
}

# ============================================================
# 完成
# ============================================================
cat("\n====================================\n")
cat("全部完成!\n")
cat(sprintf("结果保存在: %s\n", output_dir))
cat("====================================\n")

cat("\n输出文件列表:\n")
output_files <- list.files(output_dir, recursive = TRUE)
for (f in output_files) {
  cat(sprintf("  - %s\n", f))
}

cat("\n特征重要性摘要:\n")
cat(sprintf("  最重要特征: %s (%.4f)\n", 
            importance_df$feature_group[1], 
            importance_df$importance_mean[1]))
cat(sprintf("  graph_embedding 排名: %d\n", 
            importance_df$rank[importance_df$feature_group == "graph_embedding"]))
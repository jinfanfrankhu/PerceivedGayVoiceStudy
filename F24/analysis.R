# Gay Voice Perception Research — R Analysis
# Analyses: 1A (Overall Accuracy), 5D2 (PC1 vs Mean Rating)

library(tidyverse)
library(ggrepel)

INPUT_CSV  <- "Results.csv"
OUTPUT_DIR <- "NWAV Dataplots"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# ============================================================
# HELPER: choose Pearson vs Spearman via Shapiro-Wilk
# ============================================================

choose_cor <- function(x, y, alpha = 0.05) {
  # Shapiro-Wilk requires n >= 3 and n <= 5000
  sw_x <- shapiro.test(x)
  sw_y <- shapiro.test(y)
  if (sw_x$p.value < alpha || sw_y$p.value < alpha) {
    method <- "spearman"
    cat(sprintf(
      "  Normality rejected (Shapiro-Wilk: x p=%.4f, y p=%.4f) → using Spearman's rho\n",
      sw_x$p.value, sw_y$p.value
    ))
  } else {
    method <- "pearson"
    cat(sprintf(
      "  Normality not rejected (Shapiro-Wilk: x p=%.4f, y p=%.4f) → using Pearson's r\n",
      sw_x$p.value, sw_y$p.value
    ))
  }
  cor.test(x, y, method = method, exact = FALSE)
}

# ============================================================
# DATA LOADING
# ============================================================

raw <- read.csv(INPUT_CSV, header = FALSE, stringsAsFactors = FALSE)

speaker_codes <- trimws(as.character(raw[1, 2:37]))

ratings <- raw[3:87, 2:37]
ratings[] <- lapply(ratings, function(x) suppressWarnings(as.numeric(x)))
colnames(ratings) <- speaker_codes
rownames(ratings) <- NULL

demo_cols    <- c("YOB","Gender","Orientation","English","GayFriends","GayAcquaint","Religiosity","Score")
demographics <- raw[3:87, 38:45]
demographics[] <- lapply(demographics, function(x) suppressWarnings(as.numeric(x)))
colnames(demographics) <- demo_cols
rownames(demographics) <- NULL

actual_scores <- suppressWarnings(as.numeric(raw[90, 2:37]))
names(actual_scores) <- speaker_codes

avg_predicted <- suppressWarnings(as.numeric(raw[91, 2:37]))
names(avg_predicted) <- speaker_codes

cat(sprintf("Loaded %d listeners x %d speakers\n", nrow(ratings), ncol(ratings)))

# ============================================================
# 1A: OVERALL ACCURACY
# ============================================================

cat("Overall Accuracy")

df_1a <- data.frame(
  actual    = actual_scores,
  predicted = avg_predicted,
  speaker   = speaker_codes
) |> drop_na()

cat("  Normality test for 1A:\n")
res_1a <- choose_cor(df_1a$actual, df_1a$predicted)

stat_label <- if (res_1a$method == "Pearson's product-moment correlation") "r" else "\u03c1"
stat_val   <- unname(res_1a$estimate)
p_val_1a   <- res_1a$p.value

cat(sprintf("  %s = %.4f, p = %.6f\n", stat_label, stat_val, p_val_1a))

p_1a <- ggplot(df_1a, aes(x = actual, y = predicted)) +
  geom_point(color = "darkgreen", fill = "green3", shape = 21, size = 4, alpha = 0.8, stroke = 0.8) +
  geom_smooth(method = "lm", color = "red", se = FALSE, linewidth = 1.2) +
  geom_text_repel(aes(label = speaker), size = 2.8, color = "black",
                  box.padding = 0.3, max.overlaps = Inf) +
  scale_x_continuous(limits = c(1, 5), breaks = 1:5) +
  scale_y_continuous(limits = c(1, 5), breaks = 1:5) +
  labs(
    title = sprintf("Overall Perception Accuracy\n%s = %.3f, p = %.4f", stat_label, stat_val, p_val_1a),
    x = "Converted Speaker-Reported Score (1-5)",
    y = "Mean Listener-Predicted Score (1-5)"
  ) +
  theme_bw(base_size = 13) +
  theme(plot.title = element_text(hjust = 0.5))

ggsave(file.path(OUTPUT_DIR, "Overall Perception Accuracy.png"), p_1a, width = 9, height = 8, dpi = 300)
cat("  Saved Overall Perception Accuracy.png\n")

# ============================================================
# 5D2: PC1 vs MEAN RATING
# PCA on speakers x listeners (ratings transposed), then PC1
# per speaker correlated with mean listener rating
# ============================================================

cat("\n--- 5D2: PC1 vs Mean Rating ---\n")

# Impute missing per-row (speaker) with row mean, then grand mean
ratings_mat <- as.matrix(ratings)
ratings_T   <- t(ratings_mat)  # speakers x listeners (36 x 85)

# Row-wise imputation (speaker row = mean of that speaker's available ratings)
ratings_imp <- t(apply(ratings_T, 1, function(row) {
  row[is.na(row)] <- mean(row, na.rm = TRUE)
  row
}))
# Fill any remaining NAs with grand mean
grand_mean <- mean(ratings_T, na.rm = TRUE)
ratings_imp[is.na(ratings_imp)] <- grand_mean

# Standardise (scale columns = listeners)
ratings_scaled <- scale(ratings_imp)

# PCA
pca_result <- prcomp(ratings_scaled, center = FALSE, scale. = FALSE)
pc1_var    <- summary(pca_result)$importance["Proportion of Variance", "PC1"] * 100
pc1_scores <- pca_result$x[, 1]
names(pc1_scores) <- speaker_codes

df_5d2 <- data.frame(
  speaker    = speaker_codes,
  PC1        = pc1_scores,
  MeanRating = avg_predicted,
  Actual     = actual_scores
) |> drop_na()

cat("  Normality test for 5D2:\n")
res_5d2 <- choose_cor(df_5d2$MeanRating, df_5d2$PC1)

stat_label_5d2 <- if (res_5d2$method == "Pearson's product-moment correlation") "r" else "\u03c1"
stat_val_5d2   <- unname(res_5d2$estimate)
p_val_5d2      <- res_5d2$p.value

cat(sprintf("  %s = %.4f, p = %.6f\n", stat_label_5d2, stat_val_5d2, p_val_5d2))

p_5d2 <- ggplot(df_5d2, aes(x = MeanRating, y = PC1)) +
  geom_point(color = "darkgreen", fill = "mediumseagreen", shape = 21, size = 4, alpha = 0.8, stroke = 0.8) +
  geom_smooth(method = "lm", color = "red", se = FALSE, linewidth = 1.2) +
  geom_text_repel(aes(label = speaker), size = 2.8, color = "black",
                  box.padding = 0.3, max.overlaps = Inf) +
  scale_x_continuous(breaks = c(1, 2, 3, 4, 5), limits = c(1, 5)) +
  labs(
    title = sprintf("PC1 vs Mean Listener Rating\n%s = %.3f, p = %.4f", stat_label_5d2, stat_val_5d2, p_val_5d2),
    x = "Mean Listener-Predicted Score (1-5)",
    y = sprintf("PC1 Score (%.1f%% variance)", pc1_var)
  ) +
  theme_bw(base_size = 13) +
  theme(plot.title = element_text(hjust = 0.5))

ggsave(file.path(OUTPUT_DIR, "PC1 vs Mean Listener Rating.png"), p_5d2, width = 9, height = 8, dpi = 300)
cat("  Saved PC1 vs Mean Listener Rating.png\n")

# ============================================================
# 5D1: PC1 vs ACTUAL (normality check + correlation only)
# ============================================================

cat("\n--- 5D1: PC1 vs Actual Orientation ---\n")
cat("  Normality test for 5D1:\n")
res_5d1 <- choose_cor(df_5d2$Actual, df_5d2$PC1)

stat_label_5d1 <- if (res_5d1$method == "Pearson's product-moment correlation") "r" else "\u03c1"
stat_val_5d1   <- unname(res_5d1$estimate)
p_val_5d1      <- res_5d1$p.value

cat(sprintf("  %s = %.4f, p = %.6f\n", stat_label_5d1, stat_val_5d1, p_val_5d1))

cat("\nDone.\n")

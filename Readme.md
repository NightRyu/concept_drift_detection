# Adaptive Malicious IP Event Prediction under Concept Drift

Based on the UGR'16 network traffic dataset, this project investigates the impact of concept drift on the prediction of malicious traffic in dynamic network environments. Unlike traditional single-stream classification, this project reframes the task as near-future IP-level risk prediction: by utilizing feature aggregation and rolling historical features, it predicts whether a given source IP will trigger a malicious event in the near future.

`dataprocess.py` Aggregate raw flow-level data by “source IP + 1-minute window” to calculate the rate (Flow_Packets_s), average packet size, and TCP flag statistics. Simultaneously, compute rolling features for the past 10 minutes and generate a malicious prediction label (Target Label) for the next 5 minutes.

`static_baseline.py` The model is trained only once on the initial dataset, and its parameters are not updated as the remaining data streams are processed. Record and visualize the decline in performance over time.

`PSI_applied_model_2.py` & `PSI_applied_model.py` Introduce the Population Stability Index (PSI) to monitor feature drift. We have optimized the update logic so that the tree model is no longer completely rebuilt from scratch; instead, we use LightGBM’s `init_model` parameter to append tree structures to the existing model. We have also introduced a dynamic positive-negative sample weighting mechanism to address the issue of extreme class imbalance.

`double_compare.py` & `double_compare_2.py` Under a unified time-series simulation, the Static and Adaptive models are run simultaneously, and their predictions are compared. The script calculates not only the streaming F1 score but also evaluation metrics for highly challenging and highly imbalanced scenarios: global and rolling PR-AUC, as well as Recall@90% Precision.
`double_compare_2.py` further integrates an advanced version of the adaptive model and can generate visually appealing comparison plots of these challenging metrics.

---

The simulation model ultimately selected was `PSI_applied_model_2.py`

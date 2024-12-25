# Does Positionl Encoding Matter?

This project arises from lectures' discussion about properties of different positional encoding approaches, namely how
performance of transformers depends on positional encoding. To challenge this question I investigate how models learned on small context
size data perform on larger ones. 

# Predict Position (Ranking)

![Kendall by Position](./pictures/predict_position_kendall.png)

![Kendall by Position (BOS included)](./pictures/predict_position_bos_kendall.png)

# Induction 

![Accuracy by Position](./pictures/induction_acc_by_model.png)

# Addition 

# SCAN 
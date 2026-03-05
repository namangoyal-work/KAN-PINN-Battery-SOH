import torch
import matplotlib.pyplot as plt
from src.kan_pinn import pinn
from src.metrics import calculate_metrics

def evaluate_model(test_dataloader, model_path="kan_pinn_model.pth"):
    model=pinn()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predictions=[]
    ground_truths=[]
    with torch.no_grad():
        for batch in test_dataloader:
            x_features,_,u_true,_=batch
            u_hat=model(x_features)
            predictions.extend(u_hat.squeeze().tolist())
            ground_truths.extend(u_true.squeeze().tolist())
        pred_tensor=torch.tensor(predictions)
        true_tensor=torch.tensor(ground_truths)
        metrics=calculate_metrics(pred_tensor,true_tensor)

        print("--- Test Set Results ---")
        print(f"RMSPE: {metrics['RMSPE']:.3f}%")
        print(f"MAE:   {metrics['MAE']:.3f}%")
        print(f"R^2:   {metrics['R2']:.3f}")

        plt.figure(figsize=(8,6))
        plt.scatter(ground_truths,predictions,alpha=0.5,color='blue',label='Predictions')
        min_val=min(min(ground_truths),min(predictions))
        max_val=max(max(ground_truths),max(predictions))
        plt.plot([min_val,max_val],[min_val,max_val],'r--',label='Ideal Fit')
        plt.xlabel('True Capacity Loss (1 - SOH)')
        plt.ylabel('Predicted Capacity Loss')
        plt.title('AL-PKAN Model Evaluation')
        plt.legend()
        plt.grid(True)
        plt.savefig('evaluation_plot.png')
        print("Saved plot to evaluation_plot.png



if __name__ == "__main__":
              pass

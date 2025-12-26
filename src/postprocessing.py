import numpy as np
import pandas as pd

def psuedo_label(X_masked,y_masked,X_test,probs_test, n_percent = 5):

    N_SAMPLES = round(len(probs_test) * n_percent / 100)

    # to psuedo label
    idx_pos = np.argsort(probs_test)[-N_SAMPLES:] 
    idx_neg = np.argsort(probs_test)[:N_SAMPLES]

    X_pseudo_pos = X_test.iloc[idx_pos].copy()
    X_pseudo_neg = X_test.iloc[idx_neg].copy()

    y_pseudo_pos = pd.Series(1, index=X_pseudo_pos.index)
    y_pseudo_neg = pd.Series(0, index=X_pseudo_neg.index)

    print(f"Adding {len(X_pseudo_pos)} Positive and {len(X_pseudo_neg)} Negative pseudo-labels.")
    print(f"Confidences: Pos Min {probs_test[idx_pos].min():.4f} | Neg Max {probs_test[idx_neg].max():.4f}")

    # Concatenate with original
    X_final = pd.concat([X_masked, X_pseudo_pos, X_pseudo_neg])
    y_final = pd.concat([y_masked, y_pseudo_pos, y_pseudo_neg])

    return X_final, y_final

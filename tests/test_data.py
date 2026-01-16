import numpy as np
import pytest
import torch

from credit_card_fraud_analysis.data import preprocess_data, RAW_DATA_DIR
DATA_MISSING = not (RAW_DATA_DIR / "creditcard.csv").exists()


@pytest.mark.skipif(DATA_MISSING, reason="Data file 'creditcard.csv' not found")
def test_data():
    """Verify data loading, shapes, and label representation."""
    # Load processed data
    # Returns: X_train, X_test, y_train, y_test, X_train_tensor, X_test_tensor
    data = preprocess_data()
    X_train, X_test = data[0], data[1]
    y_train, y_test = data[2], data[3]
    X_train_t, X_test_t = data[4], data[5]

    # 1. Assert lengths (Expected counts for 70/30 split with SMOTE on Train)
    # Total Class 0: 284,315. 70% is 199,020. SMOTE balances Class 1 to match.
    # N_train = 199,020 * 2 = 398,040
    # N_test = 30% of original data = ~85,443
    assert len(X_train) == 398040, f"Expected 398,040 training samples, got {len(X_train)}"
    assert len(X_test) == 85443, f"Expected 85,443 test samples, got {len(X_test)}"

    # 2. Assert that each datapoint has the correct shape [28] (V1-V28 or V2-Amount)
    assert X_train.shape[1] == 28
    assert X_test.shape[1] == 28
    # Check tensor shapes specifically
    assert X_train_t.shape == (398040, 28)
    assert X_test_t.shape == (85443, 28)

    # 3. Assert that all labels (0 and 1) are represented
    # Using torch.unique as per requested style
    train_labels = torch.unique(torch.tensor(y_train))
    test_labels = torch.unique(torch.tensor(y_test))

    assert torch.equal(train_labels.sort()[0], torch.tensor([0, 1])), "Not all labels in train set"
    assert torch.equal(test_labels.sort()[0], torch.tensor([0, 1])), "Not all labels in test set"

    # 4. Verify SMOTE worked (Training set is exactly 50/50)
    assert np.count_nonzero(y_train == 0) == np.count_nonzero(y_train == 1)
# tests/test_model.py
import pytest
from s5_ci.model import MyAwesomeModel
import torch

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match='Expected each sample to have shape [1, 28, 28]'):
        model(torch.randn(1,1,28,29))
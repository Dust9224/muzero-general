import torch
from models import dict_to_cpu

def test_dict_to_cpu_tensor_moved_to_cpu():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = {
        'a': torch.tensor([1.0], device=device),
        'b': {
            'c': torch.tensor([2.0], device=device)
        }
    }
    result = dict_to_cpu(data)
    assert result['a'].device.type == 'cpu'
    assert result['b']['c'].device.type == 'cpu'


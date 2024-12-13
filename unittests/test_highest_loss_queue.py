import torch

from src.training_queues import HighestLossQueue

T = torch.tensor


def test_priority_queue_without_shuffle():
    q = HighestLossQueue([(T(1), 1), (T(2), 2), (T(3), 3)], shuffle=False)

    (x1, x2), _ = q.get_batch(2)
    q.push_batch(losses=(T(1.), T(3.)), epoch_nr=0)  # q = 3, 2, 1

    (x3, x4), _ = q.get_batch(2)
    q.push_batch(losses=(T(0.), T(2.)), epoch_nr=0)  # q = 2, 1, 3

    (x5, x6), _ = q.get_batch(2)

    elements = [x1, x2, x3, x4, x5, x6]
    elements = [tensor.item() for tensor in elements]
    assert elements == [1, 2, 3, 2, 2, 1]

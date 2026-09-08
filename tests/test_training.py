import torch
from torch.utils.data import DataLoader, Dataset

from src.data.pipeline import collate_iq
from src.models import CompactRFNet
from src.training import train_model


class ToyDataset(Dataset):
    def __init__(self, count=12):
        generator = torch.Generator().manual_seed(3)
        self.samples = torch.randn(count, 2, 16, generator=generator)
        self.labels = torch.arange(count) % 2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.samples[index], int(self.labels[index]), {"snr_db": None, "extra": "kept"}


def test_collation_preserves_unknown_metadata():
    samples, labels, metadata = next(iter(DataLoader(ToyDataset(4), batch_size=4, collate_fn=collate_iq)))
    assert samples.shape == (4, 2, 16)
    assert labels.shape == (4,)
    assert metadata[0]["snr_db"] is None


def test_training_loop_records_history_and_states():
    loader = DataLoader(ToyDataset(), batch_size=4, collate_fn=collate_iq)
    model = CompactRFNet(
        num_classes=2,
        input_length=16,
        conv_channels=(4,),
        kernel_sizes=(3,),
        pooling=(2,),
        classifier_size=4,
        dropout=0.0,
    )
    result = train_model(
        model,
        loader,
        loader,
        {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "epochs": 2,
            "early_stopping_patience": 2,
        },
        torch.device("cpu"),
    )
    assert len(result.history) == 2
    assert 1 <= result.best_epoch <= 2
    assert set(result.best_state) == set(model.state_dict())

import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaModel, GPT2ForSequenceClassification, GPT2Model

roberta_freeze_layers = [1, 4, 6]
def freeze_model_weights(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

class CustomRobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)  # LSTM layer
        self.attention = nn.MultiheadAttention(config.hidden_size, num_heads=1)  # Attention layer
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.elu = nn.ELU()

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dense1(x)
        x = self.elu(x)
        x, _ = self.lstm(x.unsqueeze(1))  # Pass through LSTM layer
        x = x.squeeze(1)
        x, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))  # Pass through Attention layer
        x = x.squeeze(0)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CustomRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        for i, layer in enumerate(self.roberta.encoder.layer):
            if i in roberta_freeze_layers:
                freeze_model_weights(layer)

        self.classifier = CustomRobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()


gpt2_freeze_layers = [1, 3, 5]

class CustomGPT2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.n_embd
        self.dense_1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.dense_2 = nn.Linear(2 * hidden_size, hidden_size)
        self.norm = nn.LayerNorm(2 * hidden_size)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.out_proj = nn.Linear(hidden_size, config.num_labels, bias=False)
        self.p_relu = nn.PReLU()

    def forward(self, x):
        x = self.dense_1(x)
        x = self.p_relu(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CustomGPT2ForSequenceClassification(GPT2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.score = CustomGPT2ClassificationHead(config)

        for i, layer in enumerate(self.transformer.h):
            if i in gpt2_freeze_layers:
                freeze_model_weights(layer)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
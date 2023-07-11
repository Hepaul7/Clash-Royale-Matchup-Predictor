import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CARDS = 109


class CardInteractionNet(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super(CardInteractionNet, self).__init__()

        # embedding layer for card representations
        self.embeddings = nn.Embedding(NUM_CARDS * 2, embedding_dim)
        # linear layer for card interaction matrix
        self.interaction_layer = nn.Linear(NUM_CARDS * NUM_CARDS, hidden_dim)
        # hidden layers
        self.fc1 = nn.Linear(embedding_dim * NUM_CARDS + hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # output layer
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, input_vectors, interaction_matrix):
        input_embedded = self.embeddings(input_vectors)
        interactions = interaction_matrix.view(-1, NUM_CARDS * NUM_CARDS)
        interaction_output = self.interaction_layer(interactions)
        combined = torch.cat((input_embedded.view(-1), interaction_output.view(-1)))
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))

        # Output layer
        output = torch.sigmoid(self.fc3(x))

        return output

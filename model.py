from imports import *


if not os.path.exists('models'):
    os.mkdir('models')

def create_model(input_size, lr=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, 4, 1, input_shape=input_size))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Conv2D(64, 4, 1))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Conv2D(128, 4, 1))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(50, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))



class GNN(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, output_dim)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
#        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
#        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
#        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv4(x, edge_index)

        return gmp(x, data.batch)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.gcn = GNN(9, 32, 19)
        self.linear1 = nn.Linear(19, 32)
        self.linear2 = nn.Linear(32, 1)

    def forward(self, graphs):
        encoded = self.gcn(graphs)

        output = self.linear1(encoded)
        output = F.relu(output)
        output = F.dropout(output, p=0.3, training=self.training)
        output = self.linear2(output)

        # Check if logit or what this is.
        # return #.squeeze(dim=1)
        return torch.sigmoid(output)


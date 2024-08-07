from utilities import *
from torch_classes import *
import torch
from torch_geometric.loader import DataLoader
from random import shuffle
from torch_geometric.utils.convert import from_networkx



def extract_graph_data(G):
    pyg_graph = from_networkx(G)
    node_feature = torch.stack((pyg_graph.weight, pyg_graph.genus, pyg_graph.disks), dim=1)
    edge_attr = pyg_graph.orientation
    edge_attr = edge_attr.view(-1, 1)
    edge_idx = pyg_graph.edge_index
    node_feature = node_feature.float()
    return node_feature, edge_idx.long(), edge_attr.float()


def DataBaseGenerator(n_equiv, n_inequiv, n_tweak, Nmax):
    data = []
    for i in range(n_equiv):
        G, G1, indicator = EquivPair(Nmax)
        node_feature_s, edge_idx_s, edge_attr_s = extract_graph_data(G)
        node_feature_t, edge_idx_t, edge_attr_t = extract_graph_data(G1)
        data.append(
            GraphPair(edge_index_s=edge_idx_s, edge_attr_s=edge_attr_s, x_s=node_feature_s,
                      edge_index_t=edge_idx_t, edge_attr_t=edge_attr_t, x_t=node_feature_t,
                      y=torch.tensor([1])))
    for i in range(n_inequiv):
            G, G1, indicator = InequivPair(Nmax)
            node_feature_s, edge_idx_s, edge_attr_s = extract_graph_data(G)
            node_feature_t, edge_idx_t, edge_attr_t = extract_graph_data(G1)
            data.append(
                GraphPair(edge_index_s=edge_idx_s, edge_attr_s=edge_attr_s, x_s=node_feature_s,
                      edge_index_t=edge_idx_t, edge_attr_t=edge_attr_t, x_t=node_feature_t,
                      y=torch.tensor([0])))
    for i in range(n_tweak):
        G, G1, indicator = TweakPair(Nmax)
        node_feature_s, edge_idx_s, edge_attr_s = extract_graph_data(G)
        node_feature_t, edge_idx_t, edge_attr_t = extract_graph_data(G1)
        data.append(
            GraphPair(edge_index_s=edge_idx_s, edge_attr_s=edge_attr_s, x_s=node_feature_s,
                      edge_index_t=edge_idx_t, edge_attr_t=edge_attr_t, x_t=node_feature_t,
                      y=torch.tensor([0])))
    return data
print("Generating the Database...")
dataset = DataBaseGenerator(40000, 20000, 20000, 60)
print("Database Generated.")
data_size = len(dataset)
idx = list(range(data_size))
shuffle(idx)
train_idx = idx[:int(data_size * 0.8)]
test_idx = idx[int(data_size * 0.8):]
train_set = [dataset[index] for index in train_idx]
test_set = [dataset[index] for index in test_idx]
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, follow_batch=['x_s', 'x_t'])
test_loader = DataLoader(test_set, batch_size=16, shuffle=True, follow_batch=['x_s', 'x_t'])
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Check if MPS is available
mps_available = torch.backends.mps.is_available()
print(f"MPS available: {mps_available}")
# If MPS is available, do not set the device to MPS as torch_scatter does not currently work on it
# device = torch.device("mps" if mps_available else "cpu")
print(f"Using device: {device}")
torch.cuda.manual_seed(seed + 3)


model_0 = NNConvGCN().to(device)
model_1 = GCNGCN().to(device)
model_2 = GCNGAT().to(device)
model_3 = NNConvNNConv().to(device)
model_4 = NNConvGAT().to(device)
model_5 = GENGAT().to(device)

criterion = torch.nn.CrossEntropyLoss()

# Start Training
print('##### Training session #####')


def train(model):
    running_loss = 0
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        x = model(data)  # Perform a single forward pass.
        loss = criterion(x, data.y)  # Compute the loss.
        running_loss += loss.item()
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
    return running_loss / len(train_loader)


def test(loader, model):
    model.eval()
    correct = 0

    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def run_epsiodes(model, ep_max, name):
    train_acc = []
    test_acc = []
    loss_buffer = []
    test_max = 0
    last_epoch = 0
    for epoch in range(0, ep_max):
        loss_buffer.append(train(model))
        trainx = test(train_loader, model)
        testx = test(test_loader, model)

        if testx > test_max:
            last_epoch=epoch
            test_max = testx
            torch.save(model.state_dict(), name + '.pth')
            print(f"****Model saved at epoch{epoch}")

        train_acc.append(trainx)
        test_acc.append(testx)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:04d}, Train Acc: {trainx:.4f}, Test Acc: {testx:.4f}')

    print('===========================================')
    print('Train ACC:')
    print(f'    Minimum {np.min(train_acc) : .4f} \n    Maximum {np.max(train_acc) : .4f} ')
    print('Test ACC:')
    print(f'    Minimum {np.min(test_acc) : .4f} \n    Maximum {np.max(test_acc) : .4f} ')
    print('===========================================')
    print('Test Average ACC:')
    print(f'    Mean {np.mean(test_acc[last_epoch:]) : .4f} \n    StDev {np.std(test_acc[last_epoch:]) : .4f} ')
    return train_acc, test_acc, loss_buffer


episodes_max = 150
learning_rate = 0.005


print('Training NNConvGCN Model...')
train_acc_0, test_acc_0, loss_buffer_0 = run_epsiodes(model_0, episodes_max, "NNConvGCN")
print('Training GCNGCN Model...')
train_acc_1, test_acc_1, loss_buffer_1 = run_epsiodes(model_1, episodes_max, "GCNGCN")
print('Training GCNGAT Model...')
train_acc_2, test_acc_2, loss_buffer_2 = run_epsiodes(model_2, episodes_max, "GCNGAT")
print('Training NNConvNNConv Model...')
train_acc_3, test_acc_3, loss_buffer_3 = run_epsiodes(model_3, episodes_max, "NNConvNNConv")
print('Training NNConvGAT Model...')
train_acc_4, test_acc_4, loss_buffer_4 = run_epsiodes(model_4, episodes_max, "NNConvGAT")
print('Training GENGAT Model...')
train_acc_5, test_acc_5, loss_buffer_5 = run_epsiodes(model_5, episodes_max, "GENGAT")

# Visualize Training Results
# Plot Loss
epoch = range(1, len(loss_buffer_1) + 1)
plt.plot(epoch, loss_buffer_0, color='g', label='NNConv+GCN')
plt.plot(epoch, loss_buffer_1, color='r', label='GCN+GCN')
plt.plot(epoch, loss_buffer_2, color='b', label='GCN+GAT')
plt.plot(epoch, loss_buffer_3, color='c', label='NNConv+NNConv')
plt.plot(epoch, loss_buffer_4, color='k', label='NNConv+GAT')
plt.plot(epoch, loss_buffer_5, color='m', label='GEN+GAT')
plt.legend(loc='best')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig('img_loss.pdf', bbox_inches='tight', format="pdf")
plt.show()
# Plot Accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Accuracy vs Epochs')

epoch = range(1, len(train_acc_1) + 1)
ax1.plot(epoch, train_acc_0, color='g', label='NNConv+GCN')
ax1.plot(epoch, train_acc_1, color='r', label='GCN+GCN')
ax1.plot(epoch, train_acc_2, color='b', label='GCN+GAT')
ax1.plot(epoch, train_acc_3, color='c', label='NNConv+NNConv')
ax1.plot(epoch, train_acc_4, color='k', label='NNConv+GAT')
ax1.plot(epoch, train_acc_5, color='m', label='GEN+GAT')
ax1.grid()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title("Train Accuracy Curve")

ax2.plot(epoch, test_acc_0, color='g', label='NNConv+GCN')
ax2.plot(epoch, test_acc_1, color='r', label='GCN+GCN')
ax2.plot(epoch, test_acc_2, color='b', label='GCN+GAT')
ax2.plot(epoch, test_acc_3, color='c', label='NNConv+NNConv')
ax2.plot(epoch, test_acc_4, color='k', label='NNConv+GAT')
ax2.plot(epoch, test_acc_5, color='m', label='GEN+GAT')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title("Validation Accuracy Curve")
plt.legend(loc='best')
ax2.grid()
plt.savefig('img_accuracy.pdf', bbox_inches='tight', format="pdf")
plt.show()

print('##### Finished Training Session #####')
print('===========================================')

import torch.optim as optim

from model.mark1 import Mark
from dataset import get_data_loader, get_online_data_loader
from option import option
from tqdm import tqdm
from loss import SimpleTrain as TrainLoss
from loss import SimpleTest as TestLoss
from test import test_function

args = option()

network = Mark(args).to(args.device)
try:
    # network.load_param()
    print('parameter load success')
except:
    print('parameter load fail')

optimizer = optim.Adam(network.parameters(), weight_decay=1e-4)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
loss_train_func = TrainLoss(args)
loss_test_func = TestLoss(args)

train_loader = get_data_loader(args)
test_loader = get_online_data_loader(args)

best_score = 0
best_epoch = 0
for epoch in range(1, args.epochs + 1):
    train_loss = 0
    network.train()
    bar = tqdm(enumerate(train_loader))
    for batch_idx, data in bar:
        data_input, data_target = data
        data_input, data_target = data_input.to(args.device), data_target.to(args.device)
        optimizer.zero_grad()
        output = network(data_input)
        loss = loss_train_func(output, data_target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            bar.set_description('Train Epoch: {:4d} [{:3d}/{:3d}]   Loss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader), loss.item()))
    train_loss /= len(train_loader)
    print('Train Epoch: {:4d} Average Train Loss: {:.6f}'.format(epoch, train_loss))
    network.save_param()
    network.eval()
    f1_score = test_function(args, network, test_loader, loss_test_func)

    if (best_score < f1_score) and (epoch > 10):
        best_score = f1_score
        best_epoch = epoch
        network.save_param('./param/best.pth')
    # scheduler.step(test_loss)
    print('f1_score : {:6f} | best_score : {:6f} | best_epoch : {:4d}\n'.format(f1_score, best_score, best_epoch))

import torch
from tqdm import tqdm


def test_function(args, network, test_loader, loss_func):
    true_positive, false_positive, false_negative = 0, 0, 0
    with torch.no_grad():
        for data in test_loader:
            data_input, data_target = data
            data_input, data_target = data_input.to(args.device), data_target.to(args.device)
            output = network(data_input)

            true_positive_, false_positive_, false_negative_ = loss_func(output, data_target)
            true_positive += true_positive_
            false_positive += false_positive_
            false_negative += false_negative_

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0
    print('Test set: Precision: {:.4f} | Recall: {:.4f} | F1 score: {:.4f}'.format(precision, recall, f1_score))
    return f1_score


if __name__ == "__main__":
    from model.mark1 import Mark
    from option import option
    from dataset import get_online_data_loader
    from loss import SimpleTest as TestLoss
    args = option()
    network = Mark(args).to(args.device)
    test_loader = get_online_data_loader(args)
    loss_func = TestLoss(args)
    try:
        network.load_param('./param/param.pth')
    except:
        pass
    network.eval()
    test_function(args, network, test_loader, loss_func)

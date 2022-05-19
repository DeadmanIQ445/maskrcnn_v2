
from dataset_fasterrcnn import TreeDataset
import torch
from references import utils


from references.engine import evaluate
import os


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load("models/RetinaNet_0.6219756949905441_95.pth")
    samples_dir = '/media/deadman445/disk/Add/test_tiles200'

    dataset_test = TreeDataset(csv_file=os.path.join(samples_dir, "final_df.csv"), root_dir=samples_dir, train=False)


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)


    eval_ret = evaluate(model, data_loader_test, device=device, epoch=1000)



if __name__ == '__main__':
    main()

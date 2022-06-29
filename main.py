import random
import torch
import numpy as np
import pandas as pd
import argparse

from sklearn.preprocessing import StandardScaler, RobustScaler
from itertools import product

from src.models import AE
from src.utils import _random_seed_gen
from src.opt import ol_filter_dkbo_one_shot

def _batched_test(x_tensor, x_init, y_init, train_times=100, batch_size:int=3, beta=5,
                    acq="ucb", verbose=False, lr=1e-2, name="test", fix_seed=False,  pretrained=False, ae_loc=None):
    '''
    Return the candidate of batched BALLET
    '''
    # fix the seed
    if fix_seed:
        _seed = int(_random_seed_gen()[0])
        torch.manual_seed(_seed)
        np.random.seed(_seed)
        random.seed(_seed)
        torch.cuda.manual_seed(_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    candidates_idx = ol_filter_dkbo_one_shot(x_tensor=x_tensor, init_x=x_init, init_y=y_init, n_iter=batch_size, fix_seed=False,
                                            train_times=train_times, acq=acq, verbose=verbose, lr=lr, name=name, beta=beta,
                                            pretrained=pretrained, ae_loc=ae_loc,)

    return candidates_idx
    

if __name__ == "__main__":
    # parse the cli
    cli_parser = argparse.ArgumentParser(description="configuration of mf test")
    cli_parser.add_argument("--name", nargs='?', default="test", type=str, help="name of experiment",)
    cli_parser.add_argument("--aedir", nargs='?', default="none", type=str, help="directory of the pretrained Autoencoder",)
    cli_parser.add_argument("--subdir", nargs='?', default="./res", type=str, help="directory to store the scripts and results",)
    cli_parser.add_argument("--datadir", nargs='?', default="./data", type=str, help="directory of the test datasets")
    cli_parser.add_argument("-a",  action="store_true", default=False, help="flag of if retrain AE")
    cli_parser.add_argument("-v",  action="store_true", default=False, help="flag of if verbose")
    cli_parser.add_argument("-f",  action="store_true", default=False, help="flag of if using fixed seed")
    cli_parser.add_argument("-n", action="store_true", default=False, help="flag of if negate the obj value to maximize")
    cli_parser.add_argument("--learning_rate", nargs='?', default=2, type=int, help="rank of the learning rate")
    cli_parser.add_argument("--train_times", nargs='?', default=100, type=int, help="number of training iterations")
    cli_parser.add_argument("--batch-size", nargs="?", default=5, type=int,help="Batch size of each round of query")    
    cli_parser.add_argument("--acq_func", nargs='?', default="ts", type=str, help="acquisition function")
    cli_parser.add_argument("--beta", nargs='?', default=2, type=float, help="confidence factor")
    
    cli_args = cli_parser.parse_args()



    # all candidates
    _biomass_categories     = ['Switchgrass', 'Hemp', 'MxG', 'Commercial Lignin']
    _biomass_categories_num = list(range(len(_biomass_categories)))
    _temp                   = list(range(600, 1600, 100))
    _fe                     = [200, 100, 50, 25, 12, 6, 3, 2, 1]
    _koh                    = [0, 20, 40, 60, 80, 100]
    _tmp = [torch.tensor(pts).reshape([1, -1]) for pts in product(_biomass_categories_num, _temp, _fe, _koh)]
    search_space = torch.cat(_tmp)
    # print(search_space, search_space.size(0))

    ### load dataset
    assert not (cli_args.datadir is None)
    if cli_args.datadir.endswith(".pt"):
        dataset = torch.load(cli_args.datadir)  # want to find the maximal
        data_dim = dataset.shape[1]-1
    elif cli_args.datadir.endswith(".npy"):
        dataset = torch.from_numpy(np.load(cli_args.datadir))
        data_dim = dataset.shape[1]-1
    elif cli_args.datadir.endswith(".xlsx"):
        _dataset = pd.read_excel(cli_args.datadir)
        _header = list(_dataset.columns)
        assert _header == ['Biomass source',
        'Highest temp. (oC)',
        'Fe load (mC%)',
        'KOH load (mC%)',
        'Capacity of Li']

        _dataset['Biomass source'].replace(_biomass_categories,
                        _biomass_categories_num, inplace=True)
        dataset = torch.tensor(_dataset[_header].values)
        data_dim = dataset.shape[1]


    # original Objective
    ROBUST = True
    ScalerClass = RobustScaler if ROBUST else StandardScaler
    scaler = ScalerClass().fit(search_space)                   
    scaled_data = scaler.transform(dataset[:,:-1])

    scaled_search_space = torch.from_numpy(scaler.transform(search_space)).float()
    scaled_input_tensor = torch.from_numpy(scaled_data).float()
    output_tensor = dataset[:,-1].float()
    shuffle_filter = np.random.choice(scaled_data.shape[0], scaled_data.shape[0], replace=False)
    train_output = -1 * output_tensor.reshape([-1,1]).float() if cli_args.n else output_tensor.reshape([-1,1]).float()

    # dkbo experiment
    verbose = cli_args.v
    fix_seed = cli_args.f
    lr_rank = -cli_args.learning_rate
    learning_rate = 10 ** lr_rank
    pretrained = not (cli_args.aedir == 'none')

    # pretrain AE
    if pretrained and cli_args.a:
        ae = AE(scaled_search_space, lr=1e-4)
        ae.train_ae(epochs=cli_args.train_times, batch_size=100, verbose=True)
        torch.save(ae.state_dict(), cli_args.aedir)
        if cli_args.v:
            print(f"pretrained ae stored in {cli_args.aedir}")

    # one shot
    print(f"Learning rate {learning_rate}")
    candidates = _batched_test(x_tensor=scaled_search_space, x_init=scaled_input_tensor, y_init=output_tensor, train_times=cli_args.train_times, batch_size=cli_args.batch_size, beta=5,
                    acq=cli_args.acq_func, verbose=verbose, lr=learning_rate, name=cli_args.name, fix_seed=fix_seed,  pretrained=pretrained, ae_loc=cli_args.aedir)

    # reverse transform
    candidates = scaler.inverse_transform(torch.cat(candidates)).round()
    candidates = pd.DataFrame(candidates, columns = _header[:-1])
    candidates['Biomass source'].replace(_biomass_categories_num, _biomass_categories, inplace=True)
    print(f"Picked Candidates:\n {candidates}")
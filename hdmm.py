import time
import copy
import argparse

from tqdm import tqdm
from scipy.sparse.linalg import lsmr
from scipy.stats import norm, laplace

from Util.data_pub_sampling import *
from Util.hdmm.workload import Marginals
from Util.hdmm import templates, error

import pdb

def get_local_strategy(W_local, local_domain, num_restarts=3):
    best_obj = np.inf
    for _ in tqdm(range(num_restarts)):

        temp = templates.Marginals(local_domain, approx=True)
        obj = temp.optimize(W_local)
        print(f'obj = {obj}')
        if obj < best_obj:
            best_obj = obj
            A = temp.strategy()
    return A

def get_l2_sen(temp_A):
    l2 = temp_A.sparse_matrix().power(2).sum(axis=0).max()
    return np.sqrt(l2)

def gaus_mech(ds, marginal_W, eps, delta):
    ds_size = len(ds.df)

    l2 = marginal_W.sparse_matrix().power(2).sum(axis=0).max()  # for spares matrices
    noise = norm(loc=0, scale=np.sqrt(l2 * 2 * np.log(2 / delta)) / eps)
    x = ds.datavector()
    z = noise.rvs(size=marginal_W.shape[0])
    a = marginal_W.dot(x)
    y = a + z
    return y / ds_size

def low_dim_hdmm(ds, marginal_W, marginal_A, eps, delta):
    ds_size = len(ds.df)
    l1 = marginal_A.sensitivity()
    l2 = 1
    if delta > 0:
        noise = norm(loc=0, scale=np.sqrt(l2 * 2 * np.log(2 / delta)) / eps)
    else:
        noise = laplace(loc=0, scale=l1 / eps)

    x = ds.datavector()
    z = noise.rvs(size=marginal_A.shape[0])
    a = marginal_A.dot(x)
    y = a + z
    ls_answer = lsmr(marginal_A, y)
    x_hat = ls_answer[0]
    ans = marginal_W.dot(x_hat)
    return ans / ds_size

def get_errors(true, est):
    diff = true - est
    max_error = np.max(np.abs(diff))
    mean_error = np.mean(np.abs(true - est))
    mse = np.mean(diff**2)
    return max_error, mean_error, mse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--dataset', type=str, help='queries', default='adult-small')
    parser.add_argument('--marginal', type=int, help='queries', default=3)
    parser.add_argument('--workload', type=int, help='queries', default=35)
    parser.add_argument('--workload_seed', type=int, default=0)
    parser.add_argument('--epsilon', type=float, help='Privacy parameter', default=0.1)
    # acs params
    parser.add_argument('--state', type=str, default=None)
    # adult params
    parser.add_argument('--adult_seed', type=int, default=None)

    args = parser.parse_args()

    print(args)
    return args

if __name__ == "__main__":
    args = get_args()

    dataset_name = args.dataset
    if args.dataset.startswith('acs_') and args.state is not None:
        dataset_name += '_{}'.format(args.state)
    if args.dataset.startswith('adult') and args.adult_seed is not None:
        dataset_name += '_{}'.format(args.adult_seed)

    results_dir ='results/{}'.format(dataset_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    proj = get_proj(args.dataset)
    if args.dataset.endswith('-small'):
        args.dataset = args.dataset[:-6]

    filter_private, filter_pub = get_filters(args)

    data, workloads = randomKway(args.dataset, args.workload, args.marginal, seed=args.workload_seed, proj=proj,
                                 filter=filter_private, args=args)
    domain = data.domain.shape
    N = len(data.df)
    delta = 1 / N ** 2

    weights = {}
    attrs = list(data.domain.config.keys())
    for workload in workloads:
        cl = tuple(map(lambda x: attrs.index(x), workload))
        weights[cl] = 1.0

    _W = Marginals.fromtuples(domain, weights)
    real_answers = _W.dot(data.datavector()) / data.df.shape[0]

    print(data.df.columns)
    print('N = {}'.format(N))
    print('# queries = {}'.format(len(real_answers)))
    print('W sensitivity = {}'.format(_W.sensitivity()))

    result_cols = {'adult_seed': [args.adult_seed],
                   'marginal': [args.marginal],
                   'num_workloads': [len(workloads)],
                   'workload_seed': [args.workload_seed],
                   'num_queries': [len(real_answers)],
                   'priv_size': [N],
                   }
    for _ in range(args.num_runs):
        run_id = hash(time.time())

        W = copy.deepcopy(_W)
        A = get_local_strategy(W, data.domain.shape)

        fake_answers = low_dim_hdmm(data, W, A, args.epsilon, delta)
        fake_answers = np.clip(fake_answers, 0, 1)
        max_error, mean_error, mse = get_errors(real_answers, fake_answers)

        results = {'run_id': run_id,
                   'epsilon': args.epsilon,
                   }
        results_errors = {'max_error': max_error,
                          'mean_error': mean_error,
                          'mean_squared_error': mse
                          }

        df_results = pd.DataFrame.from_dict(result_cols)
        for key, val in results.items():
            df_results[key] = val
        for key, val in results_errors.items():
            df_results[key] = val

        if args.dataset != 'adult':
            del df_results['adult_seed']
        results_path = os.path.join(results_dir, 'hdmm.csv')
        save_results(df_results, results_path=results_path)
        print(df_results)
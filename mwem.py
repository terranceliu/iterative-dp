import sys
sys.path.append("../private-pgm/src")

import time
import argparse
from tqdm import tqdm

import Util.util_general as util
from Util.data_pub_sampling import *
from Util.qm import QueryManager
from Util.cdp2adp import cdp_rho

"""
Our implementation of MWEM uses the Gaussian mechanism (and the zCDP composition theorem), with the option of replacing
the exponential mechanism with the Permute-and-Flip mechanism (TODO: deprecated, just use exp with factor 2 trick)
"""
def generate(data, real_answers, A_init, query_manager, N, T, eps0, permute=False, no_opt=False):
    data_onehot = None

    A = np.copy(A_init)
    A_noisy_best = np.copy(A_init)
    A_avg = np.zeros(A_init.shape)

    measurements_dict = {}

    best_score = np.infty
    for t in tqdm(range(T)):
        if query_manager.xy is not None:
            fake_answers = query_manager.get_answer_weights(A)  # saves runtime at the cost of memory
        else:
            fake_answers = query_manager.get_answer(data, weights=A)

        # 1) Exponential Mechanism
        score = np.abs(real_answers - fake_answers)
        if permute: # permute + flip
            EM_dist_0 = np.exp(eps0 * N / 2 * score, dtype=np.float128)  # Note: sensitivity is 1/N
            EM_dist = EM_dist_0 / EM_dist_0.sum()
            q_t_ind = util.sample_permute(EM_dist)
        else: # exponential mech
            """
            Lemma 3.2 of https://arxiv.org/pdf/2004.07223.pdf
            Proves that with zCDP accounting, the exponential mechanism is eps0 ^ 2 / 8 - cDP
            Saves privacy budget by a factor of 2 (TODO: update arxiv paper) 
            """
            EM_dist_0 = np.exp(2 * eps0 * N / 2 * score, dtype=np.float128)  # Note: sensitivity is 1/N
            EM_dist = EM_dist_0 / EM_dist_0.sum()
            q_t_ind = util.sample(EM_dist)

        noisy_score = score[q_t_ind]
        # print(score.max())
        if noisy_score < best_score:
            best_score = noisy_score
            A_noisy_best = np.copy(A)

        # 2) Discrete Gaussian Mechanism
        m_t = real_answers[q_t_ind]
        m_t += np.random.normal(loc=0, scale=(1 / (N * eps0)))
        m_t = np.clip(m_t, 0, 1)

        # 3) Multiplicative Weights update
        if query_manager.query_attrs is not None:
            query_attrs = query_manager.query_attrs[q_t_ind]
            query_mask = query_attrs != -1
            q_t_x = data.df.values[:, query_mask] - query_attrs[query_mask]
            q_t_x = np.abs(q_t_x).sum(axis=1)
            q_t_x = (q_t_x == 0).astype(int)
        else:
            if data_onehot is None:
                data_onehot = util.get_data_onehot(data)
            query = query_manager.get_query_workload([q_t_ind])
            q_t_x = data_onehot.dot(query.T).flatten()
            q_t_x = (q_t_x == query.sum()).astype(int)

        measurements_dict[q_t_ind] = (q_t_x, m_t)

        errors_dict = {}
        for idx, (q_t_x, m_t) in measurements_dict.items():
            q_t_A = fake_answers[idx]
            errors_dict[idx] = np.abs(m_t - q_t_A).max()
        mask = np.array(list(errors_dict.values())) >= 0.5 * errors_dict[q_t_ind]
        past_indices = np.array(list(errors_dict.keys()))[mask]

        np.random.shuffle(past_indices)
        for i in [q_t_ind] + list(past_indices):
            q_t_A = fake_answers[i]
            q_t_x, m_t = measurements_dict[i]

            factor = np.exp(q_t_x * (m_t - q_t_A) / 2)
            A = A * factor
            A = A / A.sum()

            if no_opt: # normal MWEM
                break

        A_avg += A

    A_last = np.copy(A)
    A_avg /= T

    return A_avg, A_last, A_noisy_best

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--dataset', type=str, help='queries', default='adult')
    parser.add_argument('--marginal', type=int, help='queries', default=3)
    parser.add_argument('--workload', type=int, help='queries', default=32)
    parser.add_argument('--workload_seed', type=int, default=0)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--epsilon', type=float, help='Privacy parameter', default=0.1)
    # misc params
    parser.add_argument('--permute', action='store_true')
    parser.add_argument('--no_opt', action='store_true', help='run without past queries')
    parser.add_argument('--support_size', type=int, default=None)
    parser.add_argument('--support_seed', type=int, default=0)
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
    save_dir_query = 'save/qm/{}/{}_{}_{}/'.format(dataset_name, args.marginal, args.workload, args.workload_seed)
    save_dir_xy = save_dir_query
    if args.support_size is not None:
        save_dir_xy = save_dir_xy + 'mwem/{}'.format(args.support_size)
    for d in [results_dir, save_dir_query, save_dir_xy]:
        if not os.path.exists(d):
            os.makedirs(d)

    proj = get_proj(args.dataset)
    if args.dataset.endswith('-small'):
        args.dataset = args.dataset[:-6]

    filter_private, filter_pub = get_filters(args)

    data, workloads = randomKway(args.dataset, args.workload, args.marginal, seed=args.workload_seed, proj=proj,
                                 filter=filter_private, args=args)

    query_manager = QueryManager(data.domain, workloads)
    N = data.df.shape[0]

    data_support, A_init = get_support(data)
    if args.support_size is not None:
        df_support = data_support.df
        prng = np.random.RandomState(args.support_seed)
        idxs = prng.choice(df_support.index.values, size=args.support_size, replace=False)
        df_support = df_support.loc[idxs].reset_index(drop=True)
        data_support = Dataset(df_support, data_support.domain)
        A_init = np.ones(len(df_support))
        A_init = A_init / len(A_init)

    print('workload: ', len(workloads))
    print('num queries: ', query_manager.num_queries)
    print('A:', A_init.shape)

    # get answers and initial error
    real_answers = query_manager.get_answer(data, concat=False)
    query_manager.setup_query_attr(save_dir=save_dir_query)
    query_manager.setup_xy(data_support, save_dir=save_dir_xy)
    fake_answers = query_manager.get_answer_weights(A_init, concat=False)
    init_errors = util.get_errors(real_answers, fake_answers)

    delta = 1.0 / N ** 2
    rho = cdp_rho(args.epsilon, delta)
    eps0 = (2 * rho) ** 0.5 / (2 * args.T) ** 0.5

    result_cols = {'adult_seed': [args.adult_seed],
                   'marginal': [args.marginal],
                   'num_workloads': [len(workloads)],
                   'workload_seed': [args.workload_seed],
                   'num_queries': [query_manager.num_queries],
                   'priv_size': [N],
                   'support_size': [args.support_size],
                   'support_seed': [args.support_seed]
                   }
    for _ in range(args.num_runs):
        run_id = hash(time.time())
        A_avg, A_last, A_noisy_best = generate(data_support, np.concatenate(real_answers), A_init, query_manager, N, args.T, eps0,
                                               permute=args.permute, no_opt=args.no_opt)

        fake_answers = query_manager.get_answer_weights(A_avg, concat=False)
        avg_errors = util.get_errors(real_answers, fake_answers)

        fake_answers = query_manager.get_answer_weights(A_last, concat=False)
        last_errors = util.get_errors(real_answers, fake_answers)

        fake_answers = query_manager.get_answer_weights(A_noisy_best, concat=False)
        noisy_errors = util.get_errors(real_answers, fake_answers)

        results = {'run_id': run_id,
                   'epsilon': args.epsilon,
                   'permute': args.permute,
                   'T': args.T,
                   'rho': rho,
                   'eps0': eps0,
                   }
        results_errors = {'max_error_init': init_errors['max'],
                          'mean_error_init': init_errors['mean'],
                          'median_error_init': init_errors['median'],
                          'mean_squared_error_init': init_errors['mean_squared'],
                          'mean_workload_squared_error_init': init_errors['mean_workload_squared'],

                          'max_error_last': last_errors['max'],
                          'mean_error_last': last_errors['mean'],
                          'median_error_last': last_errors['median'],
                          'mean_squared_error_last': last_errors['mean_squared'],
                          'mean_workload_squared_error_last': last_errors['mean_workload_squared'],

                          'max_error_avg': avg_errors['max'],
                          'mean_error_avg': avg_errors['mean'],
                          'median_error_avg': avg_errors['median'],
                          'mean_squared_error_avg': avg_errors['mean_squared'],
                          'mean_workload_squared_error_avg': avg_errors['mean_workload_squared'],

                          'max_error_noisy_best': noisy_errors['max'],
                          'mean_error_noisy_best': noisy_errors['mean'],
                          'median_error_noisy_best': noisy_errors['median'],
                          'mean_squared_error_noisy_best': noisy_errors['mean_squared'],
                          'mean_workload_squared_error_noisy_best': noisy_errors['mean_workload_squared'],
                          }
        df_results = pd.DataFrame.from_dict(result_cols)
        for key, val in results.items():
            df_results[key] = val
        for key, val in results_errors.items():
            df_results[key] = val

        if args.dataset != 'adult':
            del df_results['adult_seed']
        if args.support_size is None:
            del df_results['support_size']
            del df_results['support_seed']
            if args.no_opt:
                results_path = os.path.join(results_dir, 'mwem_no_opt.csv')
            else:
                results_path = os.path.join(results_dir, 'mwem.csv')
        else:
            results_path = os.path.join(results_dir, 'mwem_reduce_support.csv')
        save_results(df_results, results_path=results_path)
        print(df_results)
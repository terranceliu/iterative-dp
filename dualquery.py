import sys
sys.path.append("../private-pgm/src")

import time
import argparse
from tqdm import tqdm

from Util import oracle_dq, util_general
from Util.qm import QueryManager
from Util.data_pub_sampling import *
from Util.util_general import get_errors
from Util.cdp2adp import cdp_rho

def generate(data, query_manager, real_answers, rho_ckpts, eta, samples, mip_gap):
    domain = data.domain
    D = np.sum(domain.shape)
    N = data.df.shape[0]
    Q_size = query_manager.num_queries
    Q_dist = np.ones(2*Q_size)/(2*Q_size)

    neg_real_answers = 1 - real_answers

    rho_X_dict = {}
    rho_ckpts = list(sorted(rho_ckpts))
    rho = rho_ckpts[-1]
    accumulated_rho = 0
    X = []
    t = 0
    while True:
        # get s samples
        queries, neg_queries = [], []
        for _ in range(samples):
            # equivalent to running exp mech with privacy eps0-DP => eps0 ** 2 / 8-zCDP
            eps0 = 2 * eta * t / N
            accumulated_rho +=  eps0 ** 2 / 8
            q_id = util_general.sample(Q_dist)
            if q_id < Q_size:
                queries.append(q_id)
            else:
                neg_queries.append(q_id-Q_size)

        # Privacy consumed this round
        if accumulated_rho > rho:
            rho_X_dict[rho] = X
            break
        while(len(rho_ckpts) != 0 and accumulated_rho > rho_ckpts[0]):
            rho_X_dict[rho_ckpts.pop(0)] = X.copy()

        # Gurobi optimization: argmax_(x^t) A(x^t, q~)  >= max_x A(x, q~) - \alpha
        query_workload = query_manager.get_query_workload(queries)
        neg_query_workload = query_manager.get_query_workload(neg_queries)
        oh_fake_data = oracle_dq.dualquery_best_response(query_workload, neg_query_workload, D, domain, mip_gap)
        X.append(oh_fake_data)

        # Update query player distribution using multiplicative weights
        fake_data = Dataset(pd.DataFrame(util_general.decode_dataset(oh_fake_data, domain), columns=domain.attrs), domain)
        fake_answers = query_manager.get_answer(fake_data)
        neg_fake_answers = 1 - fake_answers
        A = np.append(real_answers - fake_answers, neg_real_answers - neg_fake_answers)
        Q_dist = np.exp(eta * A) * Q_dist

        # Normalize
        sum = np.sum(Q_dist)
        Q_dist = Q_dist / sum

        assert np.abs(np.sum(Q_dist)-1)<1e-6, "Q_dist must add up to 1"

        util_general.progress_bar(rho, accumulated_rho, msg="t={}".format(t))
        t += 1

    fake_data_dict = {}
    for eps, X in rho_X_dict.items():
        fake_data = Dataset(pd.DataFrame(util_general.decode_dataset(X, domain), columns=domain.attrs), domain)
        fake_data_dict[eps] = fake_data
    return fake_data_dict

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--dataset', type=str, help='queries', default='adult')
    parser.add_argument('--marginal', type=int, help='queries', default=3)
    parser.add_argument('--workload', type=int, help='queries', default=32)
    parser.add_argument('--workload_seed', type=int, default=0)
    parser.add_argument('--eps_ckpts', nargs='+', default=[])
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--mip_gap', type=float, default=0)
    # misc params
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
    for d in [results_dir, save_dir_query, save_dir_xy]:
        if not os.path.exists(d):
            os.makedirs(d)

    proj = get_proj(args.dataset)
    if args.dataset.endswith('-small'):
        args.dataset = args.dataset[:-6]

    filter_private, filter_pub = get_filters(args)

    data, workloads = randomKway(args.dataset, args.workload, args.marginal, seed=args.workload_seed, proj=proj,
                                 filter=filter_private, args=args)
    N = data.df.shape[0]

    query_manager = QueryManager(data.domain, workloads)
    real_answers = query_manager.get_answer(data, concat=False)

    delta = 1.0 / N ** 2
    eps_ckpts = np.array(args.eps_ckpts).astype(float)
    eps_ckpts = list(np.round(eps_ckpts, 3))
    rho_ckpts = []
    for eps in eps_ckpts:
        rho_ckpts.append(cdp_rho(eps, delta))

    result_cols = {'adult_seed': args.adult_seed,
                   'marginal': args.marginal,
                   'num_workloads': len(workloads),
                   'workload_seed': args.workload_seed,
                   'num_queries': query_manager.num_queries,
                   'priv_size': N,
                   }
    for _ in range(args.num_runs):
        run_id = hash(time.time())
        syndata_dict = generate(data=data, query_manager=query_manager, real_answers=np.concatenate(real_answers),
                                rho_ckpts=rho_ckpts, eta=args.eta, samples=args.samples, mip_gap=args.mip_gap)

        epsilons = []
        errors = {'max': [], 'mean': [], 'median': [], 'mean_squared': [], 'mean_workload_squared': []}
        max_errors, mean_errors = [], []
        for i, syndata in enumerate(syndata_dict.values()):
            fake_answers = query_manager.get_answer(syndata, concat=False)
            _errors = get_errors(real_answers, fake_answers)

            epsilons.append(eps_ckpts[i])
            for key in errors.keys():
                errors[key].append(_errors[key])

            results = {'run_id': [run_id] * len(epsilons),
                       'epsilon': epsilons,
                       'eta': [args.eta] * len(epsilons),
                       'samples': [args.samples] * len(epsilons),
                       'mip_gap': [args.mip_gap] * len(epsilons),
                       'max_error': errors['max'],
                       'mean_error': errors['mean'],
                       'median_error': errors['median'],
                       'mean_squared_error': errors['mean_squared'],
                       'mean_workload_squared_error': errors['mean_workload_squared'],
                       }
            df_results = pd.DataFrame.from_dict(results)
            i = df_results.shape[1]

            for key, val in result_cols.items():
                df_results[key] = val

            cols = list(df_results.columns[i:]) + list(df_results.columns[:i])
            df_results = df_results[cols]

            if args.dataset != 'adult':
                del df_results['adult_seed']
            results_path = os.path.join(results_dir, 'dq.csv')
            save_results(df_results, results_path=results_path)
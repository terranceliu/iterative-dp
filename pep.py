import time
import argparse
from tqdm import tqdm
from Util.hdmm.workload import Marginal

from Util.data_pub_sampling import *
from Util.cdp2adp import cdp_rho
from Util import util_general
import Util.util_general as util

import multiprocessing as mp

class MyMarginal:
    def __init__(self, domain: Domain, proj: tuple):
        self.proj = proj
        self.proj_dom = domain.project(proj)
        self.W = Marginal.fromtuple(domain.shape, domain.axes(self.proj))

    def get_W_ith_row(self, query_id):
        q = np.zeros(self.proj_dom.size())
        q[query_id] = 1
        r = self.W._transpose().dot(q)
        return r

    def get_answers_from_weights(self, weights):
        return self.W.dot(weights)

    def get_answers_from_db(self, db: Dataset):
        N = db.df.shape[0]
        return self.W.dot(db.datavector() / N)

    def sensitivity(self):
        return 1


class MyMarginals:
    def __init__(self, domain: Domain, workloads: list):
        self.workloads = workloads
        self.marginals = []
        self.query_id_to_marginal = {}
        query_id = 0
        for marginal_id, proj in enumerate(workloads):
            proj_dom = domain.project(proj)
            m = MyMarginal(domain, proj)
            self.marginals.append(m)

            for j in range(proj_dom.size()):
                self.query_id_to_marginal[query_id] = (marginal_id, j)
                query_id = query_id + 1

    def get_error(self, real: Dataset, fake: Dataset):
        return np.max(np.abs(self.get_answers(real) - self.get_answers(fake)))

    def get_error_weights(self, real_vec, fake_vec):
        return np.max(np.abs(self.get_answers_weights(real_vec) - self.get_answers_weights(fake_vec)))

    def get_answers(self, data, weights=None, concat=True, debug=False):
        ans_vec = []
        N_sync = data.df.shape[0]
        # for proj, W in self.workloads:
        for proj in self.workloads:
            # weights let's you do a weighted sum
            x = data.project(proj).datavector(weights=weights)
            if weights is None:
                x = x / N_sync
            ans_vec.append(x)
        if concat:
            ans_vec = np.concatenate(ans_vec)
        return ans_vec

    def get_answers_weights(self, weights, concat=True):
        diff = [mar.W.dot(weights) for mar in self.marginals]
        if not concat:
            return diff
        return np.concatenate(diff)


class PEP:
    def __init__(self, data_domain: Domain, my_marginals: MyMarginals, max_iters):
        # self.data_support = data_support
        self.domain_size = data_domain.size()
        self.Marginals = my_marginals
        self.max_iters = max_iters
        self.synthethic_weights = np.ones(self.domain_size) / self.domain_size

    """
    Our implementation of Iterative Projection.
    """
    def generate(self, real_db: Dataset, iterations: int, epsilon: float):
        N = len(real_db.df)
        delta = 1/N**2
        rho = cdp_rho(epsilon, delta)
        eps0 = (2 * rho) ** 0.5 / (2 * iterations) ** 0.5

        # initialize weights to zero
        self.synthethic_weights = np.ones(self.domain_size) / self.domain_size

        # get real answer
        real_workload_answers = self.Marginals.get_answers(real_db)


        self.query_measurements = []
        for t in tqdm(range(iterations)):
            # get fake answers for this round
            fake_workload_answers = self.Marginals.get_answers_weights(self.synthethic_weights)

            # 1) compute score of each workload and use the EM.
            workload_max_error_score = real_workload_answers - fake_workload_answers
            query_t_index = util_general.exponential_mechanism(np.abs(workload_max_error_score), eps0, N)

            # 2) Gaussian Mechanism with sensitivity 1/N
            real_ans_noisy = real_workload_answers[query_t_index] + np.random.normal(loc=0, scale=(1 / (N * eps0)))

            # 3) Project
            self.query_measurements.append((query_t_index, real_ans_noisy))
            self.project(t)


        return self.synthethic_weights

    def project(self, t):
        for _ in range(min(self.max_iters, 2 * (t + 1))):
            fake_answers = self.Marginals.get_answers_weights(self.synthethic_weights)

            score = [fake_answers[q_id] - temp_real for q_id, temp_real in self.query_measurements]
            i = np.argmax(np.abs(score))
            query_id, real = self.query_measurements[i]
            marginal_id, query_pos = self.Marginals.query_id_to_marginal[query_id]
            marginal = self.Marginals.marginals[marginal_id]
            self.project_one(marginal, query_pos, real, fake_answers[query_id])

    def project_one(self, marginal_query: MyMarginal, query_pos: int, real, fake):
        """
        Performs projections steps equal to 'iterations'
        """
        offset = 1e-6
        real = np.clip(real, offset, 1-offset)
        fake = np.clip(fake, offset, 1-offset)
        temp = (real * (1 - fake)) / ((1 - real) * fake)
        alpha = np.log(temp)
        x_q = marginal_query.get_W_ith_row(query_pos)
        factor = np.exp(x_q * alpha)
        self.synthethic_weights = self.synthethic_weights * factor
        self.synthethic_weights = self.synthethic_weights / self.synthethic_weights.sum()


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
    # acs params
    parser.add_argument('--state', type=str, default=None)
    # adult params
    parser.add_argument('--adult_seed', type=int, default=0)
    parser.add_argument('--iters', type=int, default=1000)

    args = parser.parse_args()

    print(args)
    return args

def main_multi_proc(AlgorithmClass):
    args = get_args()

    dataset_name = args.dataset
    if args.dataset.startswith('acs_') and args.state is not None:
        dataset_name += '_{}'.format(args.state)
    if args.dataset.startswith('adult') and args.adult_seed is not None:
        dataset_name += '_{}'.format(args.adult_seed)

    results_dir ='results/{}'.format(dataset_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    save_dir_query = 'save/qm/{}/{}_{}_{}/'.format(args.dataset, args.marginal, args.workload, args.workload_seed)
    save_dir_xy = save_dir_query
    for d in [save_dir_query, save_dir_xy]:
        if not os.path.exists(d):
            os.makedirs(d)

    proj = get_proj(args.dataset)
    if args.dataset.endswith('-small'):
        args.dataset = args.dataset[:-6]
    filter_private, filter_pub = get_filters(args)

    data, workloads = randomKway(args.dataset, args.workload, args.marginal, seed=args.workload_seed, proj=proj, filter=filter_private, args=args)
    N = data.df.shape[0]

    delta = 1.0 / N ** 2
    rho = cdp_rho(args.epsilon, delta)
    eps0 = (2 * rho) ** 0.5 / (2 * args.T) ** 0.5

    my_marginals = MyMarginals(data.domain, workloads)
    dom_size = data.domain.size()
    A_init = np.ones(dom_size) / dom_size

    # Get initial errors
    real_answers = my_marginals.get_answers(data, concat=False)
    fake_answers = my_marginals.get_answers_weights(A_init, concat=False)
    init_errors = util.get_errors(real_answers, fake_answers)

    result_cols = {'adult_seed': [args.adult_seed],
                   'marginal': [args.marginal],
                   'num_workloads': [len(workloads)],
                   'workload_seed': [args.workload_seed],
                   'num_queries': [np.concatenate(real_answers).shape[0]],
                   'priv_size': [N],
                   }
    def run_parallel(run_id, results_list):
        np.random.seed(run_id)
        ew_algorithm = AlgorithmClass(data_domain=data.domain, my_marginals=my_marginals, max_iters=args.iters)
        A_last = ew_algorithm.generate(data, args.T, args.epsilon)
        run_id = hash(time.time())
        fake_answers = my_marginals.get_answers_weights(A_last, concat=False)
        last_errors = util.get_errors(real_answers, fake_answers)

        results = {'run_id': run_id,
                   'epsilon': args.epsilon,
                   'max_iters': args.iters,
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
                          'root_mean_squared_error_last': np.sqrt(last_errors['mean_squared']),
                          'mean_workload_squared_error_last': last_errors['mean_workload_squared'],
                          }
        df_results = pd.DataFrame.from_dict(result_cols)
        for key, val in results.items():
            df_results[key] = val
        for key, val in results_errors.items():
            df_results[key] = val

        if args.dataset != 'adult':
            del df_results['adult_seed']

        results_list.append(df_results)

    processes = []
    manager = mp.Manager()
    mp_results = manager.list()
    for t in range(args.num_runs):
        proc = mp.Process(target=run_parallel, args=(t, mp_results))
        proc.start()
        processes.append(proc)

    for p in processes:
        p.join()

    for df_results in mp_results:
        print(df_results[['marginal', 'num_workloads', 'epsilon', 'T', 'max_error_last', 'root_mean_squared_error_last']])

    print(f'Saving in {results_dir}...')
    for df_results in mp_results:
        results_path = os.path.join(results_dir, '{}.csv'.format(AlgorithmClass.__name__.lower()))
        save_results(df_results, results_path=results_path)

if __name__ == "__main__":
    main_multi_proc(PEP)

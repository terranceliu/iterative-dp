import sys
sys.path.append("../private-pgm/src")

from mwem import *
from Util.data_pub_sampling import *
from Util import util_general
from Util.qm import QueryManager

class EW:
    def __init__(self, data_support, query_manager, real_answers, prev_queries_iters=1):
        self.query_manager = query_manager
        self.real_answers = real_answers
        self.data_support = data_support
        self.domain_size = len(data_support.df)
        self.iterations = prev_queries_iters
        # print(f'domain size = {self.domain_size}')
        self.synthethic_data_distribution = np.ones(self.domain_size)/ self.domain_size
        ## helper
        self.data_onehot = None

    def get_answer_vector(self, query_id):
        if self.query_manager.query_attrs is not None:
            query_attrs = self.query_manager.query_attrs[query_id]
            query_mask = query_attrs != -1
            answers_to_query_id = self.data_support.df.values[:, query_mask] - query_attrs[query_mask]
            answers_to_query_id = np.abs(answers_to_query_id).sum(axis=1)
            answers_to_query_id = (answers_to_query_id == 0).astype(int)
        else:
            if self.data_onehot is None:
                self.data_onehot = util_general.get_data_onehot(self.data_support)
            query = query_manager.get_query_workload([query_id])
            answers_to_query_id = self.data_onehot.dot(query.T).flatten()
            answers_to_query_id = (answers_to_query_id == query.sum()).astype(int)
        return answers_to_query_id

    def get_answer(self, query_ids: list):
        """
        returns q(D)
        """
        answers = np.zeros(len(query_ids))
        for i, query_id in enumerate(query_ids):
            query_at_X = self.get_answer_vector(query_id)  # q(X)
            answers[i] = np.dot(query_at_X, self.synthethic_data_distribution)
        return answers

    def project(self, noisy_answers: list, query_ids: list, iterations=1):
        """
        Performs projections steps equal to 'iterations'
        """
        offset = 1e-6
        iterations = max(iterations, 1)
        assert len(noisy_answers) == len(query_ids)
        for _ in range(iterations):
            score = np.abs(noisy_answers - self.get_answer(query_ids))
            max_query_position = np.argmax(score)
            query_id = query_ids[max_query_position]
            real = np.clip(noisy_answers[max_query_position], offset, 1-offset)

            query_answer_vector = self.get_answer_vector(query_id)
            fake_answer = np.dot(query_answer_vector, self.synthethic_data_distribution)
            fake = np.clip(fake_answer, offset, 1-offset)

            temp = (real * (1 - fake)) / ((1-real) * fake)
            alpha = np.log(temp)
            factor = np.exp(query_answer_vector * alpha)
            self.synthethic_data_distribution = self.synthethic_data_distribution * factor
            self.synthethic_data_distribution = self.synthethic_data_distribution / self.synthethic_data_distribution.sum()


    """
    Our implementation of Iterative Projection.
    """
    def generate(self, N, T, permute=False):
        offset = 1/N**2
        # lap_error_bound = np.log(T/0.10) / (eps0 * N)
        # lap_error_bound = 1 / (eps0 * N)
        lap_error_bound = 0
        # print(f'Lap error bound = {lap_error_bound}')

        rho = cdp_rho(args.epsilon, delta)
        eps0 = (2 * rho) ** 0.5 / (2 * T) ** 0.5

        best_score = np.infty
        error_list = []

        previous_queries = []
        previous_answers = []

        self.synthethic_data_distribution = np.ones(self.domain_size)/ self.domain_size

        for _ in tqdm(range(T)):
            if self.query_manager.xy is not None:
                fake_answers = self.query_manager.get_answer_weights(self.synthethic_data_distribution) # saves runtime at the cost of memory
            else:
                fake_answers = self.query_manager.get_answer(self.data_support, weights=self.synthethic_data_distribution)

            # 1) Exponential Mechanism
            score = self.real_answers - fake_answers
            query_t_index = util_general.exponential_mechanism(np.abs(score), eps0, N)
            error_list.append(np.max(np.abs(score)))

            if np.abs(score[query_t_index])< best_score:
                best_score = np.abs(score[query_t_index])

            # 2) Discrete Gaussian Mechanism
            query_t_real_ans_with_noise = self.real_answers[query_t_index] + np.random.normal(loc=0, scale=(1 / (N * eps0)))
            query_t_real_ans_with_noise += -lap_error_bound if score[query_t_index] > 0 else lap_error_bound
            query_t_real_ans_with_noise = np.clip(query_t_real_ans_with_noise, offset, 1-offset)

            # Use previous queries
            if self.iterations <= 1:
                previous_queries.clear()
                previous_answers.clear()

            previous_queries.append(query_t_index)
            previous_answers.append(query_t_real_ans_with_noise)

            # 3) Project
            self.project(previous_answers, previous_queries, iterations=self.iterations)

        return self.synthethic_data_distribution

    def generate_non_adaptive(self, N, T, permute=False):
        offset = 1/N**2
        queries = []
        answers = []
        remaining_q_ids = [i for i in range(self.real_answers.shape[0])]

        self.synthethic_data_distribution = np.ones(self.domain_size)/ self.domain_size

        for _ in tqdm(range(T)):
            if self.query_manager.xy is not None:
                fake_answers = self.query_manager.get_answer_weights(self.synthethic_data_distribution) # saves runtime at the cost of memory
            else:
                fake_answers = self.query_manager.get_answer(self.data_support, weights=self.synthethic_data_distribution)

            # 1) Exponential Mechanism
            score = self.real_answers - fake_answers
            score_rem = score[remaining_q_ids]
            # print(score_rem)
            i = util_general.exponential_mechanism(np.abs(score_rem), eps0, N)

            query_t_index = remaining_q_ids[i]
            remaining_q_ids.remove(query_t_index)


            # 2) Discrete Gaussian Mechanism
            query_t_real_ans_with_noise = self.real_answers[query_t_index] + np.random.normal(loc=0, scale=(1 / (N * eps0)))
            # query_t_real_ans_with_noise += -lap_error_bound if score[query_t_index] > 0 else lap_error_bound
            query_t_real_ans_with_noise = np.clip(query_t_real_ans_with_noise, offset, 1-offset)

            print(f'i  = {i}, query_id_t = {query_t_index}, a_t = {query_t_real_ans_with_noise} ')
            queries.append(query_t_index)
            answers.append(query_t_real_ans_with_noise)

        self.project(answers, queries, iterations=50)
        return self.synthethic_data_distribution

def get_args():
    parser = argparse.ArgumentParser()

    # privacy params
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--dataset', type=str, help='queries', default='adult')
    parser.add_argument('--marginal', type=int, help='queries', default=3)
    parser.add_argument('--workload', type=int, help='queries', default=128)
    parser.add_argument('--workload_seed', type=int, default=0)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--epsilon', type=float, help='Privacy parameter', default=0.1)
    # public dataset params
    parser.add_argument('--dataset_pub', type=str, default=None)
    parser.add_argument('--pub_frac', type=float, default=1.0)
    parser.add_argument('--frac_seed', type=int, default=0)
    # misc params
    parser.add_argument('--permute', action='store_true')
    parser.add_argument('--use_support_only', action='store_true')
    # acs params
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--state_pub', type=str, default=None)
    parser.add_argument('--iters', type=int, default=1)

    args = parser.parse_args()
    if args.dataset_pub is None:
        args.dataset_pub = args.dataset

    # validate params
    if args.dataset == 'adult':
        print("For ADULT experiments, please run pmw_pub_bias.py with the argument perturb=0")
        exit()
    if args.pub_frac == 1 and args.frac_seed != 0:
        print("Only need to run frac_seed=0 for pub_frac=1.0")
        exit()
    if args.dataset.startswith('acs_'):
        assert(args.state is not None)
        assert(args.state_pub is not None)

    print(args)
    return args

if __name__ == "__main__":
    args = get_args()

    dataset_name = args.dataset
    if args.dataset.startswith('acs_') and args.state is not None:
        dataset_name += '_{}'.format(args.state)
    results_dir ='results/{}'.format(dataset_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    save_dir_query = 'save/qm/{}/{}_{}_{}/'.format(args.dataset, args.marginal, args.workload, args.workload_seed)
    save_dir_xy = save_dir_query + 'pep_pub/{}_{}_{}_{}'.format(args.dataset_pub, args.state_pub, args.pub_frac, args.frac_seed)
    for d in [save_dir_query, save_dir_xy]:
        if not os.path.exists(d):
            os.makedirs(d)

    proj = get_proj(args.dataset)
    if args.dataset.endswith('-small'):
        if args.dataset.startswith('acs'):
            args.dataset = args.dataset[:-6]
            args.dataset_pub = args.dataset_pub[:-6]

    filter_private, filter_pub = get_filters(args)

    data, workloads = randomKway(args.dataset, args.workload, args.marginal, seed=args.workload_seed, proj=proj, filter=filter_private)
    query_manager = QueryManager(data.domain, workloads)
    N = data.df.shape[0]

    data_pub, _ = randomKway(args.dataset_pub, args.workload, args.marginal, seed=args.workload_seed, proj=proj, filter=filter_pub)
    N_pub = int(args.pub_frac * data_pub.df.shape[0])
    data_pub, A_init = get_pub_dataset(data_pub, args.pub_frac, args.frac_seed)

    if args.use_support_only:
        A_init = np.ones(A_init.shape)
        A_init = A_init / len(A_init)

    print('workload: ', len(workloads))
    print('num queries: ', query_manager.num_queries)
    print('A:', A_init.shape)

    # get answers and initial error
    real_answers = query_manager.get_answer(data, concat=False)
    query_manager.setup_query_attr(save_dir=save_dir_query)
    query_manager.setup_xy(data_pub, save_dir=save_dir_xy)
    fake_answers = query_manager.get_answer_weights(A_init, concat=False)
    init_errors = util.get_errors(real_answers, fake_answers)

    delta = 1.0 / N ** 2
    rho = cdp_rho(args.epsilon, delta)
    eps0 = (2 * rho) ** 0.5 / (2 * args.T) ** 0.5

    result_cols = {'marginal': [args.marginal],
                   'num_workloads': [len(workloads)],
                   'workload_seed': [args.workload_seed],
                   'num_queries': [query_manager.num_queries],
                   'dataset_pub': [args.dataset_pub],
                   'state_pub': [args.state_pub],
                   'pub_frac': [args.pub_frac],
                   'frac_seed': [args.frac_seed],
                   'priv_size': [N],
                   'pub_size': [N_pub],
                   }

    ew_algorithm = EW(data_support=data_pub, query_manager=query_manager, real_answers= np.concatenate(real_answers),
                      prev_queries_iters=args.iters)
    for _ in range(args.num_runs):
        run_id = hash(time.time())
        A_last = ew_algorithm.generate(N, args.T, permute=args.permute)


        fake_answers = query_manager.get_answer_weights(A_last, concat=False)
        last_errors = util.get_errors(real_answers, fake_answers)

        results = {'run_id': run_id,
                   'epsilon': args.epsilon,
                   'permute': args.permute,
                   'max_iters': args.iters,
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
                          }
        df_results = pd.DataFrame.from_dict(result_cols)
        for key, val in results.items():
            df_results[key] = val
        for key, val in results_errors.items():
            df_results[key] = val

        # only need these cols for ACS experiments
        if args.state_pub is None:
            del df_results['dataset_pub']
            del df_results['state_pub']

        # save results
        results_path = os.path.join(results_dir, 'pep_pub.csv')
        if args.use_support_only:
            results_path = os.path.join(results_dir, 'pep_pub_support.csv')
        save_results(df_results, results_path=results_path)

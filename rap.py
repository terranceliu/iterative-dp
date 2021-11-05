import time
import copy
import pickle
import argparse

from torch import optim
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, Embedding

import Util.util_general as util
from Util.util_gem import *
from Util.qm import QueryManager

def report_noisy_max(score, rho, sensitivity=1, ignore_idxs=None):
    x = score.copy()
    if ignore_idxs is not None:
        x[ignore_idxs] = -10000
    noise = np.random.laplace(loc=0, scale=(2 *sensitivity) / (2 * rho) ** 0.5, size=score.shape)
    x += noise
    return x.argmax()

def get_rho(epsilon, delta):
    a = -np.log(1 / delta) ** 0.5
    b = (np.log(1 / delta) + epsilon) ** 0.5
    x = np.maximum(a+b, a-b)
    rho = x ** 2
    assert(x >= 0)
    assert(np.isclose(epsilon, rho + 2 * (rho * np.log(1 / delta)) ** 0.5))
    return rho

def convert_answers(fake_answers, real_answers):
    workload_sizes = [len(x) for x in real_answers]
    workload_idxs = np.cumsum(workload_sizes)
    workload_idxs = np.concatenate([[0], workload_idxs])
    workload_idxs = np.vstack([workload_idxs[:-1], workload_idxs[1:]]).T

    new_fake_answers = []
    for i in range(workload_idxs.shape[0]):
        idxs = workload_idxs[i]
        x = fake_answers[idxs[0]:idxs[1]]
        new_fake_answers.append(x)

    return new_fake_answers

class UnifInitializer(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            module.weight.data = (torch.rand(w.shape) - 0.5) * (1 / 0.5)

class WeightClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1, 1)
            module.weight.data = w

class RAP(torch.nn.Module):
    def __init__(self, d, n, softmax=True):
        super(RAP, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.softmax = softmax

        self.syndata = Embedding(d, n)
        self.syndata.apply(UnifInitializer())
        self.syndata = self.syndata.to(self.device)
        self.clipper = WeightClipper()

        self.past_query_idxs = torch.tensor([])
        self.past_neg_query_mask = torch.tensor([])
        self.past_measurements = torch.tensor([])

        self.true_max_error = []

    def setup_domain(self, domain):
        self.domain = domain

        x = domain.shape
        x = np.cumsum(x)
        x = np.concatenate(([0], x))
        x = np.stack([x[:-1], x[1:]]).T
        self.query_attr_bin = x

        self.query_attr_dict = {}
        for i in np.arange(self.query_attr_bin.max()):
            self.query_attr_dict[i] = np.argmax([(i >= _x[0] and i < _x[1]) for _x in x])

    def _get_probs(self, x):
        data_t = []
        for i in range(self.query_attr_bin.shape[0]):
            bin = self.query_attr_bin[i]
            logits = x[:, bin[0]:bin[1]]
            probs = logits.softmax(-1)
            # probs = torch.clamp(probs, 0, 1e10)
            # probs = probs / probs.sum(-1).unsqueeze(-1)
            data_t.append(probs)
        return torch.cat(data_t, dim=1)

    def get_answers(self, q_t_idxs):
        x = self.syndata.weight.T
        if self.softmax:
            x = self._get_probs(x)
        x = x[:, q_t_idxs]
        x = x.prod(dim=-1)
        x = x.mean(dim=0)
        return x

    def get_answers_all(self):
        out = []
        _x = self.syndata.weight.T.detach()
        if self.softmax:
            _x = self._get_probs(_x)
        for queries in torch.split(self.queries, 10000):
            x = _x[:, queries]
            x = x.prod(dim=-1)
            x = x.mean(dim=0)
            out.append(x)
        out = torch.cat(out)
        return out

    def fit(self, T, K, rho, qm, real_answers, sensitivity=1, lr=1e-3,
            max_iters=5000, verbose=False):
        eps0 = rho / (2 * T * K)
        real_answers = torch.tensor(real_answers).to(self.device)
        queries = torch.tensor(qm.queries).to(self.device).long()
        self.queries = queries

        fake_answers = self.get_answers_all()
        answer_diffs = real_answers - fake_answers

        for t in tqdm(range(T)):
            for _ in range(K):
                score = answer_diffs.abs().cpu().numpy()

                ignore_idxs = self.past_query_idxs.cpu().numpy().astype(int)
                # max_query_idx = report_noisy_max(score, eps0, sensitivity=sensitivity, ignore_idxs=ignore_idxs)
                score[ignore_idxs] = -10000
                EM_dist_0 = np.exp((8 * eps0) ** 0.5 * score / (2 * sensitivity), dtype=np.float128)  # https://arxiv.org/pdf/2004.07223.pdf Lemma 3.2
                EM_dist = EM_dist_0 / EM_dist_0.sum()
                max_query_idx = util.sample(EM_dist)

                max_query_idx = torch.tensor([max_query_idx]).to(self.device)

                # get noisy measurements
                real_answer = real_answers[max_query_idx]
                real_answer += np.random.normal(loc=0, scale=sensitivity / (2 * eps0) ** 0.5)
                real_answer = torch.clamp(real_answer, 0, 1)

                # keep track of past queries
                if len(self.past_query_idxs) == 0:
                    self.past_query_idxs = torch.cat([max_query_idx])
                    self.past_measurements = torch.cat([real_answer])
                elif max_query_idx not in self.past_query_idxs:
                    self.past_query_idxs = torch.cat((self.past_query_idxs, max_query_idx)).clone()
                    self.past_measurements = torch.cat((self.past_measurements, real_answer)).clone()

            optimizer = optim.Adam(self.syndata.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max_iters,
                                                             threshold=1e-7, threshold_mode='rel')

            noisy_max_error = (self.past_measurements - fake_answers[self.past_query_idxs]).abs().max().item()

            step = 0
            while step < max_iters:
                step += 1
                optimizer.zero_grad()

                q_t_idxs = queries[self.past_query_idxs]
                fake_answers = self.get_answers(q_t_idxs)
                errors = self.past_measurements - fake_answers
                norm_p = 2
                loss = torch.norm(errors, p=norm_p) ** norm_p
                if loss < 1e-8:
                    break

                loss.backward()
                scheduler.step(loss)
                if scheduler.num_bad_epochs > 5:
                    break

                optimizer.step()
                if not self.softmax:
                    self.syndata.apply(self.clipper)
                # print(loss)

            fake_answers = self.get_answers_all()
            answer_diffs = real_answers - fake_answers
            true_max_error = answer_diffs.abs().max().item()

            self.true_max_error.append(answer_diffs.abs().max().item())

            if verbose:
                print("Iter {}\tError: {:.4f}\tSteps: {}\tLoss: {}".format(t, true_max_error, step, loss.item()))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--dataset', type=str, help='queries', default='adult')
    parser.add_argument('--marginal', type=int, help='queries', default=3)
    parser.add_argument('--workload', type=int, help='queries', default=32)
    parser.add_argument('--workload_seed', type=int, default=0)
    parser.add_argument('--epsilon', type=float, help='Privacy parameter', default=1.0)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--T_total', type=int, default=None)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--n', type=int, default=1000)
    # misc params
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--all_marginals', action='store_true')
    # acs params
    parser.add_argument('--state', type=str, default=None)
    # adult params
    parser.add_argument('--adult_seed', type=int, default=None)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--softmax', action='store_true')

    args = parser.parse_args()

    if args.T_total is not None:
        args.T = int(args.T_total / args.K)

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
    save_dir_query = 'save/qm/{}/{}_{}_{}/'.format(args.dataset, args.marginal, args.workload, args.workload_seed)
    for d in [results_dir, save_dir_query]:
        if not os.path.exists(d):
            os.makedirs(d)

    ### Setup Data ###
    proj = get_proj(args.dataset)
    if args.dataset.endswith('-small'):
        args.dataset = args.dataset[:-6]

    filter_private, filter_pub = get_filters(args)

    marginals = [args.marginal]
    if args.all_marginals:
        marginals += list(np.arange(args.marginal)[1:][::-1])

    workloads = []
    for marginal in marginals:
        data, _workloads = randomKway(args.dataset, args.workload, args.marginal, seed=args.workload_seed,
                                     proj=proj, filter=filter_private, args=args)
        workloads += _workloads

    N = data.df.shape[0]
    domain_dtype = data.df.max().dtype

    query_manager = QueryManager(data.domain, workloads)
    real_answers = query_manager.get_answer(data, concat=False)

    ### Train generator ###
    delta = 1.0 / N ** 2
    rho = get_rho(args.epsilon, delta)

    result_cols = {'adult_seed': args.adult_seed,
                   'marginal': args.marginal,
                   'all_marginals': args.all_marginals,
                   'num_workloads': len(workloads),
                   'workload_seed': args.workload_seed,
                   'num_queries': query_manager.num_queries,
                   'priv_size': N,
                   'lr': args.lr,
                   'epsilon': args.epsilon,
                   'T': args.T,
                   'K': args.K,
                   'rho': rho,
                   }

    errors = {'max': [], 'mean': [], 'median': [], 'mean_squared': [], 'mean_workload_squared': []}

    run_ids = []
    for _ in range(args.num_runs):
        run_id = hash(time.time())
        run_ids.append(run_id)

        rap = RAP(d=query_manager.dim, n=args.n, softmax=args.softmax)
        rap.setup_domain(data.domain)
        rap.fit(T=args.T, K=args.K, rho=rho, qm=query_manager,
                real_answers=np.concatenate(real_answers), sensitivity=1 / N,
                lr=args.lr, verbose=args.verbose)

        fake_answers = rap.get_answers_all().cpu().numpy().astype(float)
        fake_answers = np.clip(fake_answers, 0, 1)
        fake_answers = convert_answers(fake_answers, real_answers)

        _errors = util.get_errors(real_answers, fake_answers)
        for key in errors.keys():
            errors[key].append(_errors[key])

    results = {'run_id': run_ids,
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

    # rearrange columns for better presentation
    cols = list(df_results.columns[i:]) + list(df_results.columns[:i])
    df_results = df_results[cols]

    if args.dataset != 'adult':
        del df_results['adult_seed']

    if args.softmax:
        results_path = os.path.join(results_dir, 'rap_softmax.csv')
    else:
        results_path = os.path.join(results_dir, 'rap.csv')
    save_results(df_results, results_path=results_path)



























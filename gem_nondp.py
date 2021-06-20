import sys
from gem import *

class GEM_nondp(GEM):
    def fit(self, lr=1e-4, eta_min=1e-5,
            qm=None, real_answers=None, resample=False,
            max_epochs=10000, max_idxs=100, max_iters=100,
            early_stopping=500, patience=100,
            verbose=False):

        df = pd.DataFrame(columns=['epoch', 'max_error', 'iters_since_improvement', 'lr'])
        save_path = os.path.join(self.save_dir, 'best.pkl')
        log_path = os.path.join(self.save_dir, 'log.csv')

        real_answers = torch.tensor(real_answers).to(self.device)
        queries = torch.tensor(qm.queries).to(self.device).long()

        self.optimizerG = optim.Adam(self.generator.parameters(), lr=lr)
        self.schedulerG = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, factor=0.5,
                                                               mode='min', patience=patience, min_lr=1e-6)

        self.past_query_idxs = torch.tensor([])
        self.past_measurements = torch.tensor([])

        fake_data = self.generate_fake_data(self.mean, self.std, resample=resample)
        fake_answers = self._get_fake_answers(fake_data, qm)
        answer_diffs = real_answers - fake_answers

        # nondp
        best_max_error = np.infty
        iters_since_improvement = 0

        t = 0
        while(t < max_epochs):
            t += 1

            score = answer_diffs.abs().cpu().numpy()
            true_max_error = score.max().item()
            max_query_idx = score.argmax()
            max_query_idx = torch.tensor([max_query_idx]).to(self.device)
            real_answer = real_answers[max_query_idx]

            # keep track of past queries
            if len(self.past_query_idxs) == 0:
                self.past_query_idxs = torch.cat([max_query_idx])
                self.past_measurements = torch.cat([real_answer])
            elif max_query_idx not in self.past_query_idxs:
                self.past_query_idxs = torch.cat((self.past_query_idxs, max_query_idx)).clone()
                self.past_measurements = torch.cat((self.past_measurements, real_answer)).clone()

            errors, q_t_idxs = self._get_past_errors(fake_data, queries)
            THRESHOLD = 0.5 * true_max_error

            lr = None
            for param_group in self.optimizerG.param_groups:
                lr = param_group['lr']
            optimizer = optim.Adam(self.generator.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters, eta_min=1e-8)

            step = 0
            while step < max_iters:
                optimizer.zero_grad()

                idxs = torch.arange(q_t_idxs.shape[0])

                # above THRESHOLD
                mask = errors >= THRESHOLD
                idxs = idxs[mask]
                q_t_idxs = q_t_idxs[mask]
                errors = errors[mask]

                # get top MAX_IDXS
                max_errors_idxs = errors.argsort()[-max_idxs:]
                idxs = idxs[max_errors_idxs]
                q_t_idxs = q_t_idxs[max_errors_idxs]
                errors = errors[max_errors_idxs]

                if len(q_t_idxs) == 0: # no errors above threshold
                    break

                fake_query_attr = fake_data[:, queries[q_t_idxs]]
                fake_answer = fake_query_attr.prod(-1).mean(axis=0)
                real_answer = self.past_measurements[idxs].clone()

                errors = (real_answer - fake_answer).abs()
                loss = errors.mean()

                loss.backward()
                optimizer.step()
                scheduler.step()

                # generate new data for next iteration
                fake_data = self.generate_fake_data(self.mean, self.std, resample=resample)
                errors, q_t_idxs = self._get_past_errors(fake_data, queries)

                step += 1

            fake_answers = self._get_fake_answers(fake_data, qm)
            answer_diffs = real_answers - fake_answers
            true_max_error = answer_diffs.abs().max().item()

            if true_max_error < 0.001:
                return

            self.schedulerG.step(true_max_error)

            if true_max_error < best_max_error:
                best_max_error = true_max_error
                self.save(save_path)
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1

            if verbose:
                print_statement = "Epoch {}: error: {:.4f}, since improve: {}, count: {}, lr: {:.8f}".format(
                    t, true_max_error, iters_since_improvement, step, lr)
                print(print_statement, file=sys.stderr)

            row = [t, true_max_error, iters_since_improvement, lr]
            df.loc[df.shape[0]] = row
            df.to_csv(log_path, index=False)

            if iters_since_improvement > early_stopping:
                return

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='queries', default='adult')
    parser.add_argument('--marginal', type=int, help='queries', default=3)
    parser.add_argument('--workload', type=int, help='queries', default=32)
    parser.add_argument('--workload_seed', type=int, default=0)
    parser.add_argument('--all_marginals', action='store_true')
    # acs params
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--dataset_pub', type=str, default=None)
    parser.add_argument('--state_pub', type=str, default=None)
    # adult params
    parser.add_argument('--adult_seed', type=int, default=None)
    # GEM params
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--syndata_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eta_min', type=float, default=None)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--max_idxs', type=int, default=100)
    parser.add_argument('--resample', action='store_true')
    # misc params
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--reduce_attr', action='store_true')

    args = parser.parse_args()

    print(args)
    return args

if __name__ == "__main__":
    args = get_args()

    dataset_name = args.dataset
    if args.dataset.startswith('acs_') and args.state is not None:
        dataset_name += '_{}'.format(args.state)
    elif args.dataset.startswith('adult') and args.adult_seed is not None:
        dataset_name += '_{}'.format(args.adult_seed)
    if args.reduce_attr:
        dataset_name += '_reduce_attr'

    save_dir_query = 'save/qm/{}/{}_{}_{}/'.format(args.dataset, args.marginal, args.workload, args.workload_seed)
    save_dir = 'save/gem_nondp/{}/{}_{}_{}/'.format(dataset_name, args.marginal, args.workload, args.workload_seed)
    for d in [save_dir_query, save_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    ### Setup Data ###
    proj = get_proj(args.dataset)
    if args.dataset.endswith('-small'):
        args.dataset = args.dataset[:-6]

    filter_private, filter_pub = get_filters(args)
    if filter_pub[1] is None: # TODO: maybe make this cleaner later (disprepancy between loading pub data for ADULT vs ACS data)
        filter_pub = filter_private

    marginals = [args.marginal]
    if args.all_marginals:
        marginals += list(np.arange(args.marginal)[1:][::-1])

    workloads = []
    for marginal in marginals:
        data, _workloads = randomKway(args.dataset, args.workload, args.marginal, seed=args.workload_seed, proj=proj, filter=filter_pub, args=args)
        workloads += _workloads

    # hard code the reduced ADULT dataset columns
    if args.reduce_attr:
        attr_reduce = ['sex',  'race',  'relationship', 'marital-status',
                       'occupation', 'education-num',
                       'age_10',
                       ]

        # TODO: not really needed, but just to be safe for now
        attr_other = [col for col in data.df.columns if col not in attr_reduce]
        data.df.loc[:, attr_other] = 0

        workloads_array = np.array(workloads)
        mask = np.zeros(len(workloads_array)).astype(int)
        for i in range(len(attr_reduce)):
            _mask = (workloads_array == attr_reduce[i]).any(axis=-1).astype(int)
            mask += _mask
        mask = mask == args.marginal
        workloads = list(workloads_array[mask])
        workloads = [tuple(workload) for workload in workloads]

    N = data.df.shape[0]
    domain_dtype = data.df.max().dtype

    query_manager = QueryManager(data.domain, workloads)
    real_answers = query_manager.get_answer(data, concat=False)

    result_cols = {'adult_seed': args.adult_seed,
                   'marginal': args.marginal,
                   'all_marginals': args.all_marginals,
                   'num_workloads': len(workloads),
                   'workload_seed': args.workload_seed,
                   'num_queries': query_manager.num_queries,
                   'priv_size': N,
                   }
    run_id = hash(time.time())
    gem = GEM_nondp(embedding_dim=args.dim, gen_dim=[args.dim * 2, args.dim * 2], batch_size=args.syndata_size, save_dir=save_dir)

    save_path = os.path.join(save_dir, 'best.pkl')
    if os.path.exists(save_path):
        if args.continue_training:
            gem.load(save_path)
        elif args.overwrite:
            gem.setup_data(data.df, proj, data.domain)
        else:
            print("Error: Saved model exists. Please choose to either overwrite file or continue training.")
            exit()
    else:
        gem.setup_data(data.df, proj, data.domain)

    gem.fit(lr=args.lr, eta_min=args.eta_min,
            qm=query_manager, real_answers=np.concatenate(real_answers), resample=args.resample,
            max_idxs=args.max_idxs, max_iters=args.max_iters,
            verbose=args.verbose)

    num_samples = 100000 // args.syndata_size
    _errors, _errors_distr = get_syndata_errors(gem, query_manager, num_samples, data.domain, real_answers, resample=args.resample)
    for key, val in _errors.items():
        print("{}: {}".format(key, val))

    for key, val in _errors_distr.items():
        print("distr_{}: {}".format(key, val))
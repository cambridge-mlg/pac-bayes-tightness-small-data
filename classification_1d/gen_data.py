import argparse
from pacbayes.args_utils import load_data_gen


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('--input_dim',
                        type=int,
                        default=1,
                        help='Dimension of GP inputs.')
    parser.add_argument('--data',
                        choices=['eq',
                                 'matern',
                                 'noisy-mixture',
                                 'weakly-periodic',
                                 'sawtooth'],
                        default='eq',
                        help='Data set to train the CNP on. ')
    parser.add_argument('--continuous_output',
                        action='store_true',
                        help='Generate data with continuous outputs.')
    parser.add_argument('--class_scheme',
                        choices=['standard',
                                 'balanced'],
                        default='standard',
                        help='Whether to used standard or balanced '
                             'data generator. ')
    parser.add_argument('--noisy',
                        action='store_true',
                        help='Whether to add aleatoric noise to GP samples.')
    parser.add_argument('--stretch',
                        type=float,
                        default=0.7,
                        help='Stretch factor of lengthscale of kernel.')
    parser.add_argument('--scale',
                        type=float,
                        default=1.,
                        help='Scale of kernel.')
    parser.add_argument('--num_context',
                        type=int,
                        default=30,
                        help='Number of points to use in context set at train '
                             'and test time')
    parser.add_argument('--num_test',
                        type=int,
                        default=300,
                        help='Number of points used to estimate generalization'
                             ' risk')
    parser.add_argument('--name',
                        type=str,
                        help='Name of file containing saved data.')
    parser.add_argument('--num_train_batches',
                        type=int,
                        default=5000,
                        help='Number of batches of tasks for train set.')
    parser.add_argument('--num_test_batches',
                        type=int,
                        default=16,
                        help='Number of batches of tasks for test set.')

    args = parser.parse_args()

    gen = load_data_gen(args,
                        num_batches=args.num_train_batches,
                        max_train_points=args.num_context,
                        max_test_points=0)
    gen_test = load_data_gen(args,
                             num_batches=args.num_test_batches,
                             max_train_points=args.num_context,
                             max_test_points=args.num_test)

    gen.save(args.name + '_train', num_tasks=args.num_train_batches)
    gen_test.save(args.name + '_test', num_tasks=args.num_test_batches)

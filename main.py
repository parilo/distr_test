import argparse
from typing import List

import torch as t

from distr_test.cond_real_nvp import ConditionedRealNVP
from distr_test.logger import ValueLogger
from distr_test.plot import plot, plot_histogram
from distr_test.distr import DeterministicDist, NormalDist, BaseDist, GaussianMixtureDist, RealNVP
from distr_test.distr_optimizer import DistMSEOptimizer, DistrOptimizer, DistLogPOptimizer
from language_model_rl.mlp import MLP


def parse_args():
    parser = argparse.ArgumentParser(description='Perform training')
    parser.add_argument(
        '--train-steps', '--st',
        default=1,
        type=int,
        help='Number of train steps per epoch'
    )
    parser.add_argument(
        '--learning-rate', '--lr',
        default=1e-4,
        type=float,
        help='Number of train steps per epoch'
    )
    parser.add_argument(
        '--batch-size', '--bs',
        default=64,
        type=int,
        help='Number of train steps per epoch'
    )
    # parser.add_argument(
    #     '--tb-log-dir', '--tb',
    #     default=None,
    #     type=str,
    #     help='Tensorboard log dir'
    # )

    return parser.parse_args()


def test_distr_learning(
        src_distr: BaseDist,
        dst_distr: BaseDist,
        distr_optimizer: DistrOptimizer,
        train_steps: int,
        src_dist_dim: int = 1,
        batch_size: int = 64,
):
    logger = ValueLogger()

    for st_ind in range(train_steps):
        batch = src_distr((batch_size, src_dist_dim)).detach()
        log_data = distr_optimizer.train_step(batch)
        logger.log(st_ind, log_data, src_distr, dst_distr)

    return logger


def test_src_determ_dst_determ_mse(args):
    src_distr = DeterministicDist(1.2345)
    dst_distr = DeterministicDist(2.3456)
    distr_optimizer = DistMSEOptimizer(dst_distr)
    test_distr_learning(
        src_distr=src_distr,
        dst_distr=dst_distr,
        distr_optimizer=distr_optimizer,
        train_steps=args.train_steps
    )


def test_src_normal_dst_determ_mse(args):
    src_distr = NormalDist(1.2345, 2.3456)
    dst_distr = DeterministicDist(2.3456)
    distr_optimizer = DistMSEOptimizer(dst_distr)
    test_distr_learning(
        src_distr=src_distr,
        dst_distr=dst_distr,
        distr_optimizer=distr_optimizer,
        train_steps=args.train_steps
    )


def test_src_determ_dst_normal_mse(args):
    src_distr = DeterministicDist(1.2345)
    dst_distr = NormalDist(2.3456, 3.4567)
    distr_optimizer = DistMSEOptimizer(dst_distr)
    test_distr_learning(
        src_distr=src_distr,
        dst_distr=dst_distr,
        distr_optimizer=distr_optimizer,
        train_steps=args.train_steps
    )


def test_src_normal_dst_normal_mse(args):
    src_distr = NormalDist(1.2345, 2.3456)
    dst_distr = NormalDist(2.3456, 3.4567)
    distr_optimizer = DistMSEOptimizer(dst_distr)
    test_distr_learning(
        src_distr=src_distr,
        dst_distr=dst_distr,
        distr_optimizer=distr_optimizer,
        train_steps=args.train_steps
    )


def test_src_normal_dst_normal_logp(args):
    src_distr = NormalDist(1.2345, 2.3456)
    dst_distr = NormalDist(2.3456, 3.4567)
    distr_optimizer = DistLogPOptimizer(dst_distr)
    test_distr_learning(
        src_distr=src_distr,
        dst_distr=dst_distr,
        distr_optimizer=distr_optimizer,
        train_steps=args.train_steps
    )


def test_src_gmm_dst_normal_mse(args):
    src_distr = GaussianMixtureDist(1, 2, 2, 3)
    dst_distr = NormalDist(2, 2.5)
    distr_optimizer = DistMSEOptimizer(dst_distr)
    logger = test_distr_learning(
        src_distr=src_distr,
        dst_distr=dst_distr,
        distr_optimizer=distr_optimizer,
        train_steps=args.train_steps
    )
    plot(*logger.get_plot_data({
        'Loss': ['mse_loss'],
        'Mu': [
            'src gaussian mixture mu',
            'src gaussian mixture mu2',
            'dst normal mu'
        ],
        'Sigma': [
            'src gaussian mixture sigma',
            'src gaussian mixture sigma2',
            'dst normal sigma'
        ]
    }))


def test_src_gmm_dst_normal_logp(args):
    src_distr = GaussianMixtureDist(1, 2, 2, 3)
    dst_distr = NormalDist(2, 2.5)
    distr_optimizer = DistLogPOptimizer(dst_distr)
    logger = test_distr_learning(
        src_distr=src_distr,
        dst_distr=dst_distr,
        distr_optimizer=distr_optimizer,
        train_steps=args.train_steps
    )
    plot(*logger.get_plot_data({
        'Loss': ['logp_loss'],
        'Mu': [
            'src gaussian mixture mu',
            'src gaussian mixture mu2',
            'dst normal mu'
        ],
        'Sigma': [
            'src gaussian mixture sigma',
            'src gaussian mixture sigma2',
            'dst normal sigma'
        ]
    }))


def test_src_normal_dst_gmm_logp(args):
    src_distr = NormalDist(1, 1.2)
    dst_distr = GaussianMixtureDist(0.5, 1, 2, 3)
    distr_optimizer = DistLogPOptimizer(dst_distr)
    logger = test_distr_learning(
        src_distr=src_distr,
        dst_distr=dst_distr,
        distr_optimizer=distr_optimizer,
        train_steps=args.train_steps
    )

    src_samples = src_distr((10000,)).detach().numpy()
    dst_samples = dst_distr((10000,)).detach().numpy()

    plot(
        *logger.get_plot_data({
            'Loss': ['logp_loss'],
            'Mu': [
                'dst gaussian mixture mu',
                'dst gaussian mixture mu2',
                'src normal mu'
            ],
            'Sigma': [
                'dst gaussian mixture sigma',
                'dst gaussian mixture sigma2',
                'src normal sigma'
            ]
        }),
        histogram_data=[src_samples, dst_samples],
        histogram_titles=['src normal dist', 'dst gmm dist'],
    )


def test_src_gmm_dst_gmm_logp(args):
    src_distr = GaussianMixtureDist(-2, 1.2, 2, 1)
    dst_distr = GaussianMixtureDist(0.5, 0.5, 2, 2)
    distr_optimizer = DistLogPOptimizer(dst_distr)
    logger = test_distr_learning(
        src_distr=src_distr,
        dst_distr=dst_distr,
        distr_optimizer=distr_optimizer,
        train_steps=args.train_steps
    )

    src_samples = src_distr((10000,)).detach().numpy()
    dst_samples = dst_distr((10000,)).detach().numpy()

    plot(
        *logger.get_plot_data({
            'Loss': ['logp_loss'],
            'Mu': [
                'src gaussian mixture mu',
                'src gaussian mixture mu2',
                'dst gaussian mixture mu',
                'dst gaussian mixture mu2',
            ],
            'Sigma': [
                'src gaussian mixture sigma',
                'src gaussian mixture sigma2',
                'dst gaussian mixture sigma',
                'dst gaussian mixture sigma2',
            ]
        }),
        histogram_data=[src_samples, dst_samples],
        histogram_titles=['src gmm dist', 'dst gmm dist'],
    )


def test_src_real_nvp_dst_real_nvp_logp(args):
    src_dist = RealNVP(dim=2)
    dst_dist = RealNVP(dim=2)

    dist_optimizer = DistLogPOptimizer(dst_dist)
    logger = test_distr_learning(
        src_distr=src_dist,
        dst_distr=dst_dist,
        distr_optimizer=dist_optimizer,
        train_steps=args.train_steps
    )

    src_samples = src_dist((10000,)).detach().numpy()[:, 0]
    dst_samples = dst_dist((10000,)).detach().numpy()[:, 0]

    plot(
        *logger.get_plot_data({
            'Loss': ['logp_loss'],
            'Mu': [
                'src gaussian mixture mu',
                'src gaussian mixture mu2',
                'dst gaussian mixture mu',
                'dst gaussian mixture mu2',
            ],
            'Sigma': [
                'src gaussian mixture sigma',
                'src gaussian mixture sigma2',
                'dst gaussian mixture sigma',
                'dst gaussian mixture sigma2',
            ]
        }),
        histogram_data=[src_samples, dst_samples],
        histogram_titles=['src real nvp dist', 'dst real nvp dist'],
    )


def test_src_gmm_dst_real_nvp_logp(args):
    src_dist = GaussianMixtureDist([-4, 2, 10, 16], [1.2, 1, 0.8, 1.5])
    dst_dist = RealNVP(dim=2, hidden_layer_size=8)

    dist_optimizer = DistLogPOptimizer(dst_dist)
    logger = test_distr_learning(
        src_distr=src_dist,
        dst_distr=dst_dist,
        distr_optimizer=dist_optimizer,
        train_steps=args.train_steps,
        src_dist_dim=2,
    )

    src_samples = src_dist((10000,)).detach().numpy()
    dst_samples = dst_dist((10000,)).detach().numpy()[:, 0]

    plot(
        *logger.get_plot_data({
            'Loss': ['logp_loss'],
            'Mu': [
                'src gaussian mixture mu',
                'src gaussian mixture mu2',
                'dst gaussian mixture mu',
                'dst gaussian mixture mu2',
            ],
            'Sigma': [
                'src gaussian mixture sigma',
                'src gaussian mixture sigma2',
                'dst gaussian mixture sigma',
                'dst gaussian mixture sigma2',
            ]
        }),
        histogram_data=[src_samples, dst_samples],
        histogram_titles=['src gmm dist', 'dst real nvp dist'],
    )


def test_conditioned_distr_learning(
        src_distrs: List[BaseDist],
        dst_distr: BaseDist,
        conditions: List[t.Tensor],
        distr_optimizer: DistrOptimizer,
        train_steps: int,
        dst_dist_dim: int = 1,
        batch_size: int = 64,
):
    logger = ValueLogger()

    for st_ind in range(train_steps):
        for src_distr, condition in zip(src_distrs, conditions):
            batch = src_distr((batch_size, 1)).repeat(1, dst_dist_dim).detach()
            log_data = distr_optimizer.train_step(
                batch,
                condition.repeat(batch_size, 1)
            )
            logger.log(st_ind, log_data, src_distr, dst_distr)

    return logger


def test_src_gmm_dst_cond_real_nvp_logp(args):
    src_1_dist = GaussianMixtureDist([-4, 2, 10], [1.2, 1, 0.8])
    src_2_dist = GaussianMixtureDist([-2, 1, 5], [1, 0.6, 0.8])
    dst_dist = ConditionedRealNVP(
        dim=2,
        num_transforms=19,
        s_module=MLP(
            input_size=3,   # distr dim // 2 + condition size
            layers_num=4,
            layer_size=16,
            output_size=1,  # distr dim // 2
        ),
        t_module=MLP(
            input_size=3,  # distr dim // 2 + condition size
            layers_num=4,
            layer_size=16,
            output_size=1,  # distr dim // 2
        )
    )

    conditions = [
        t.zeros((2,), dtype=t.float32),
        t.ones((2,), dtype=t.float32),
    ]

    dist_optimizer = DistLogPOptimizer(dst_dist, lr=args.learning_rate)
    logger = test_conditioned_distr_learning(
        src_distrs=[src_1_dist, src_2_dist],
        dst_distr=dst_dist,
        conditions=conditions,
        distr_optimizer=dist_optimizer,
        train_steps=args.train_steps,
        dst_dist_dim=2,
        batch_size=args.batch_size
    )

    src_1_samples = src_1_dist((10000,)).detach().numpy()
    src_2_samples = src_2_dist((10000,)).detach().numpy()
    dst_cond_1_samples = dst_dist((10000,), conditions[0].repeat(10000, 1)).detach().numpy()[:, 0]
    dst_cond_2_samples = dst_dist((10000,), conditions[1].repeat(10000, 1)).detach().numpy()[:, 0]

    plot(
        *logger.get_plot_data({
            'Loss': ['logp_loss'],
            'Mu': [
                'src gaussian mixture mu',
                'src gaussian mixture mu2',
                'dst gaussian mixture mu',
                'dst gaussian mixture mu2',
            ],
            'Sigma': [
                'src gaussian mixture sigma',
                'src gaussian mixture sigma2',
                'dst gaussian mixture sigma',
                'dst gaussian mixture sigma2',
            ]
        }),
        histogram_data=[
            src_1_samples,
            src_2_samples,
            dst_cond_1_samples,
            dst_cond_2_samples
        ],
        histogram_titles=[
            'src 1 gmm dist',
            'src 2 gmm dist',
            'dst cond 1 real nvp dist',
            'dst cond 2 real nvp dist',
        ],
    )


def main():

    args = parse_args()
    # test_src_determ_dst_determ_mse(args)
    # test_src_normal_dst_determ_mse(args)
    # test_src_determ_dst_normal_mse(args)
    # test_src_normal_dst_normal_mse(args)
    # test_src_normal_dst_normal_logp(args)
    # test_src_gmm_dst_normal_mse(args)
    # test_src_gmm_dst_normal_logp(args)
    # test_src_normal_dst_gmm_logp(args)
    # test_src_gmm_dst_gmm_logp(args)
    # test_src_gmm_dst_real_nvp_logp(args)
    # test_src_real_nvp_dst_real_nvp_logp(args)
    test_src_gmm_dst_cond_real_nvp_logp(args)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

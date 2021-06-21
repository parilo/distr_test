import argparse

from logger import ValueLogger
from plot import plot
from distr import DeterministicDist, NormalDist, BaseDist, GaussianMixtureDist
from distr_optimizer import DistMSEOptimizer, DistrOptimizer, DistLogPOptimizer


def parse_args():
    parser = argparse.ArgumentParser(description='Perform training')
    parser.add_argument(
        '--train-steps', '--st',
        default=1,
        type=int,
        help='Number of train steps per epoch'
    )
    parser.add_argument(
        '--tb-log-dir', '--tb',
        default=None,
        type=str,
        help='Tensorboard log dir'
    )

    return parser.parse_args()


def test_distr_learning(
        src_distr: BaseDist,
        dst_distr: BaseDist,
        distr_optimizer: DistrOptimizer,
        train_steps: int,
):
    logger = ValueLogger()

    for st_ind in range(train_steps):
        batch = src_distr((64,))
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
    src_distr = NormalDist(1, 2)
    dst_distr = GaussianMixtureDist(0.5, 1, 2, 3)
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
            'dst gaussian mixture mu',
            'dst gaussian mixture mu2',
            'src normal mu'
        ],
        'Sigma': [
            'dst gaussian mixture sigma',
            'dst gaussian mixture sigma2',
            'src normal sigma'
        ]
    }))


def test_src_gmm_dst_gmm_logp(args):
    src_distr = GaussianMixtureDist(-0.5, 3, -1.5, 2)
    dst_distr = GaussianMixtureDist(0.5, 1, 2, 1.5)
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
            'dst gaussian mixture mu',
            'dst gaussian mixture mu2',
        ],
        'Sigma': [
            'src gaussian mixture sigma',
            'src gaussian mixture sigma2',
            'dst gaussian mixture sigma',
            'dst gaussian mixture sigma2',
        ]
    }))


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
    test_src_gmm_dst_gmm_logp(args)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

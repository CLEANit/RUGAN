from .base_options import BaseOptions


class TrainOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--beta1', type=float, default=0.0, help='beta1 of adam')
        parser.add_argument('--beta2', type=float, default=0.9, help='beta2 of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--wd', type=float, default=0.0, help='weight decay for generator')
        parser.add_argument('--num_epochs', type=int, default=2000, help='number of epochs to train')
        parser.add_argument('--gradient_penalty', type = float, default = 10, help = 'gradient penalty hyperparameter')
        parser.add_argument('--critic_iters', type = int, default = 20, help = 'number of critic iterations before generator iteration')
        parser.add_argument('--epoch', type=int, default=0, help='load model from this epoch')
        self.isTrain = True

        return parser

import argparse
import utils
from loadNet import MNIST_net
from loadNet import FashionMNIST_net
from extractSubNet import MNIST_SubNet, FashionMNIST_SubNet
from dataset import MNISTdata, MNISTdata_subset, FashionMNISTdata, FashionMNISTdata_subset


def parse_args():
    desc = "Keras implementation of 'SubNetwork'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', required = True, type=str, help='MNIST or FashionMNIST')
    parser.add_argument('--subset', required=True, nargs='+', type=int, help='subset of total categories')
    parser.add_argument('--meanNodes', default=False, required=False, help='whether you want the mean #nodes')

    return check_args(parser.parse_args())


def check_args(args):

    try:
        assert args.dataset == 'MNIST' or args.dataset == 'FashionMNIST'
    except:
        print('\nError:')
        print('Choose MNIST or FashionMNIST.')
        return None

    try:
        assert type(args.subset) == list
        assert (2 <= len(args.subset)) and (len(args.subset) <= 10)
    except:
        print('\nError:')
        print('There must be at least two categories and not more than ten categories.')
        return None

    return args


def main(args):
    if args.dataset == 'MNIST':
        print("\n'Using MNISTdataset'\n")
        model = MNIST_net()
        #test_images, test_labels = MNISTdata(test=True)
        #print('\nevaluate MNISTmodel')
        #print(model.evaluate(test_images, test_labels, verbose=0))

        # extract OriginalNet
        original_Net, items = MNIST_SubNet(model, [0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9], meanNodes=args.meanNodes)
        test_images, test_labels = MNISTdata_subset('OriginalNet', args.subset)
        print('\n<Evaluate> OriginalNet for', args.subset)
        Loss, Accuracy = original_Net.evaluate(test_images, test_labels, verbose=0)
        print('Loss: %.4f' % Loss)
        print('Accuracy: %.4f' % Accuracy)
        print('# params: %d' % original_Net.count_params())

        # extract subNet
        subNet, _ = MNIST_SubNet(model, args.subset)
        test_images, test_labels = MNISTdata_subset('SubNet', args.subset)
        print('\n<Evaluate> subNet for', args.subset)
        Loss, Accuracy = subNet.evaluate(test_images, test_labels, verbose=0)
        print('Loss: %.4f' % Loss)
        print('Accuracy: %.4f' % Accuracy)
        print('# params: %d' % subNet.count_params())

        if args.meanNodes:
            print("\n<Mean of Nodes>")
            utils.calMeanNodes(items)

    if args.dataset == 'FashionMNIST':
        print("\n'Using FashionMNISTdataset'\n")
        model = FashionMNIST_net()
        #test_images, test_labels = FashionMNISTdata(test=True)
        #print('\nevaluate FashionMNISTmodel')
        #print(model.evaluate(test_images, test_labels, verbose=0))

        # extract OriginalNet
        original_Net, items = FashionMNIST_SubNet(model, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], meanNodes=args.meanNodes)
        test_images, test_labels = FashionMNISTdata_subset('OriginalNet', args.subset)
        print('\n<Evaluate> OriginalNet for', args.subset)
        Loss, Accuracy = original_Net.evaluate(test_images, test_labels, verbose=0)
        print('Loss: %.4f' % Loss)
        print('Accuracy: %.4f' % Accuracy)
        print('# params: %d' %original_Net.count_params())

        # extract subNet
        subNet, _ = FashionMNIST_SubNet(model, args.subset)
        test_images, test_labels = FashionMNISTdata_subset('SubNet', args.subset)
        print('\n<Evaluate> subNet for', args.subset)
        Loss, Accuracy = subNet.evaluate(test_images, test_labels, verbose=0)
        print('Loss: %.4f'%Loss)
        print('Accuracy: %.4f'% Accuracy)
        print('# params: %d'%subNet.count_params())

        if args.meanNodes:
            print("\n<Mean of Nodes>")
            utils.calMeanNodes(items)


if __name__ == '__main__':
    args = parse_args()
    if args is None:
        exit()
    main(args)
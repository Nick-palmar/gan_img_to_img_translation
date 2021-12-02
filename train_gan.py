
from data_utility import show_batch, Data
import os


def main():
    data = Data('apples', 'oranges', False, 64, (128, 128))
    data.get_loaders('apples_and_oranges')
    show_batch(data, 9)
    # show_batch(data.dlTargetTrain)
    


main()

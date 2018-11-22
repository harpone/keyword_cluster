import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from os.path import join, isfile
import ast


def main(path_data=None,
         out_path=None):
    #path_root = '/mnt/TERA/Data/reddit_topics'
    #path_data = join(path_root, 'safe_links_all')
    #out_path = join(path_root, 'img_reddits.csv')

    image_types = ['.jpg', '.png']
    df = []
    with open(path_data, 'r') as f:
        for n, line in enumerate(f):
            # print(f'\r{n}', end='')
            try:
                lst = ast.literal_eval(line)  # [subreddit, submission title, submitted link, comments link, short name]
            except ValueError:
                # print('ValueError')
                continue
            if any([x in lst[2] for x in image_types]):
                df.append(lst)
            if len(df) > 1000000:  # memory efficient
                df = pd.DataFrame(df)
                df.columns = ['subreddit', 'submission_title', 'submission_link', 'comments_link', 'short_name']
                if not isfile(out_path):
                    df.to_csv(out_path, index=False)
                else:
                    df.to_csv(out_path, mode='a', header=False, index=False)
                df = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vocabulary for the resnet model')
    parser.add_argument('-path_data',
                        help='Path to the dataset ("safe_links_all")',
                        default='./safe_links_all')
    parser.add_argument('-out_path',
                        help='Output path where the processed dataset will be saved',
                        default='./img_reddits.csv')
    args = parser.parse_args()

    main(**vars(args))


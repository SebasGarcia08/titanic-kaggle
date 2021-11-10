import os
from tqdm import tqdm


def download_data(output_dir: str = '.'):
    """Downloads the data from the internet."""
    print("Downloading data...")
    repo_url = 'https://raw.githubusercontent.com/rebeccabilbro/titanic/master/data'
    files = ['data_dictionary.txt', 'train.csv', 'test.csv']
    pbar = tqdm(files)
    for f in pbar:
        pbar.set_description(f'Downloading {f}...')
        os.system(f'wget -P {output_dir} {repo_url}/{f}')
    print("Done.")


if __name__ == '__main__':
    download_data()

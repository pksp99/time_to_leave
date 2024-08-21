import pandas as pd

from utils import methods, generate_data

def main(input:str, n:int):
    df = pd.read_csv(methods.file_path(input))
    total = df['n'] + df['N']
    df['n'] = n
    df['N'] = total - df['n']
    generate_data.generate(config=dict(),
                           output=str(n)+input,
                           log_file='app.log',
                           df=df)


if __name__ == '__main__':
    main('gen_test.csv', 8)
    main('gen_train.csv', 8)

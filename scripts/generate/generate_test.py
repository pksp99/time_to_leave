from utils import generate_data, config


def main():
    generate_data.generate(config.TEST_CONFIG,
                           'gen_test.csv',
                           'app.log',
                           n=5000)


if __name__ == '__main__':
    main()
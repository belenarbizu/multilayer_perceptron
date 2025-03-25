from toolkit import DataParser
import matplotlib.pyplot as plt

def histogram(data):
    plt.hist(data.iloc[:, 2])
    plt.show()

def main():
    data = DataParser.open_file("data.csv", 0)
    histogram(data)

if __name__ == "__main__":
    main()